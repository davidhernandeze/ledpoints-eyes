/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl'
import * as mpHands from '@mediapipe/hands'

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm'

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
    tfjsWasm.version_wasm}/dist/`)

import * as handdetection from '@tensorflow-models/hand-pose-detection'

import {Camera} from './camera'
import {setupDatGui} from './option_panel'
import {STATE} from './shared/params'
import {setBackendAndEnvFlags} from './shared/util'

let detector, camera
let rafId

async function createDetector() {
  switch (STATE.model) {
    case handdetection.SupportedModels.MediaPipeHands:
      const runtime = STATE.backend.split('-')[0]
      if (runtime === 'mediapipe') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: 1,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}`,
        })
      } else if (runtime === 'tfjs') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands,
        })
      }
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera)
    STATE.isTargetFPSChanged = false
    STATE.isSizeOptionChanged = false
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true

    window.cancelAnimationFrame(rafId)

    if (detector != null) {
      detector.dispose()
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend)
    }

    try {
      detector = await createDetector(STATE.model)
    } catch (error) {
      detector = null
      alert(error)
    }

    STATE.isFlagChanged = false
    STATE.isBackendChanged = false
    STATE.isModelChanged = false
  }
}

class DurationTimer {
  constructor(duration, callback) {
    this.elapsed = 0
    this.timer = null
    this.duration = duration
    this.callback = callback
    this.running = false
  }

  start() {
    if (this.running) return
    this.running = true
    this.timer = setInterval(() => {
      this.elapsed += 100
      setProgress(Math.round((this.elapsed / this.duration) * 100) / 100)
      if (this.elapsed >= this.duration) {
        this.callback()
        this.stop()
      }
    }, 100)
  }

  stop() {
    this.running = false
    this.elapsed = 0
    setProgress(0)
    clearInterval(this.timer)
  }
}

let isInCommandMode = false

const openPoseTimer = new DurationTimer(1000, () => {
  isInCommandMode = true
  document.getElementById('open').style.display = 'block'
})


const commandModeTimer = new DurationTimer(1200, () => {
  isInCommandMode = false
  document.getElementById('open').style.display = 'none'
})

const optionOnePoseTimer = new DurationTimer(500, (extendedFingers) => {
  isInCommandMode = false
  document.getElementById('team1').innerText = parseInt(document.getElementById('team1').innerText) + 1
  document.getElementById('open').style.display = 'none'
})

const optionTwoPoseTimer = new DurationTimer(500, (extendedFingers) => {
  isInCommandMode = false
  document.getElementById('team2').innerText = parseInt(document.getElementById('team2').innerText) + 1
  document.getElementById('open').style.display = 'none'
})

const optionThreePoseTimer = new DurationTimer(500, (extendedFingers) => {
  isInCommandMode = false
  document.getElementById('team3').innerText = parseInt(document.getElementById('team3').innerText) + 1
  document.getElementById('open').style.display = 'none'
})


async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video)
      }
    })
  }

  let hands = null

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateHands.

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      hands = await detector.estimateHands(
        camera.video,
        {flipHorizontal: false})
    } catch (error) {
      detector.dispose()
      detector = null
      alert(error)
    }

    let isHandOpen = false

    if (hands.length > 0) {
      const handLandmarks = hands[0].keypoints
      const extendedFingers = getExtendedFingers(handLandmarks)
      isHandOpen = extendedFingers === 4

      if (isInCommandMode) {
        if (extendedFingers === 0 || extendedFingers > 4) {
          optionOnePoseTimer.stop()
          optionTwoPoseTimer.stop()
          optionThreePoseTimer.stop()
        } else if (extendedFingers === 1) {
          optionOnePoseTimer.start()
          optionTwoPoseTimer.stop()
          optionThreePoseTimer.stop()
        } else if (extendedFingers === 2) {
          optionTwoPoseTimer.start()
          optionOnePoseTimer.stop()
          optionThreePoseTimer.stop()
        } else if (extendedFingers === 3) {
          optionThreePoseTimer.start()
          optionOnePoseTimer.stop()
          optionTwoPoseTimer.stop()
        }
      }
    }

    if (isHandOpen) {
      if (!isInCommandMode) openPoseTimer.start()
    }

    if (!isHandOpen) {
      openPoseTimer.stop()
      if (isInCommandMode) {
        commandModeTimer.start()
      }
    }
  }

  camera.drawCtx()

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (hands && hands.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(hands)
  }
}

async function renderPrediction() {
  await checkGuiUpdate()

  if (!STATE.isModelChanged) {
    await renderResult()
  }

  rafId = requestAnimationFrame(renderPrediction)
};

async function app() {
  const urlParams = new URLSearchParams(window.location.search)
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.')
    return
  }

  await setupDatGui(urlParams)

  camera = await Camera.setupCamera(STATE.camera)

  await setBackendAndEnvFlags(STATE.flags, STATE.backend)

  detector = await createDetector()

  renderPrediction()
}

function calculate2DAngle(pointA, pointB, pointC) {
  const AB = {x: pointB.x - pointA.x, y: pointB.y - pointA.y}
  const BC = {x: pointC.x - pointB.x, y: pointC.y - pointB.y}
  const dotProduct = AB.x * BC.x + AB.y * BC.y
  const magAB = Math.sqrt(AB.x * AB.x + AB.y * AB.y)
  const magBC = Math.sqrt(BC.x * BC.x + BC.y * BC.y)
  if (magAB === 0 || magBC === 0) {
    return null
  }
  const cosTheta = dotProduct / (magAB * magBC)
  return Math.acos(cosTheta) * (180 / Math.PI)
}

function isFingerExtended(mcp, pip, tip) {
  const angleMCP_PIP_TIP = calculate2DAngle(mcp, pip, tip)
  const straightThreshold = 20
  return angleMCP_PIP_TIP < straightThreshold && mcp.y > tip.y && (mcp.y - tip.y) > 10
}

function isHandFullyOpen(landmarks) {
  const extendedFingers = getExtendedFingers(landmarks)
  return extendedFingers === 4
}

function getExtendedFingers(landmarks) {
  const indexMCP = landmarks[5], indexPIP = landmarks[6], indexTip = landmarks[8]
  const middleMCP = landmarks[9], middlePIP = landmarks[10], middleTip = landmarks[12]
  const ringMCP = landmarks[13], ringPIP = landmarks[14], ringTip = landmarks[16]
  const pinkyMCP = landmarks[17], pinkyPIP = landmarks[18], pinkyTip = landmarks[20]

  const indexExtended = isFingerExtended(indexMCP, indexPIP, indexTip)
  const middleExtended = isFingerExtended(middleMCP, middlePIP, middleTip)
  const ringExtended = isFingerExtended(ringMCP, ringPIP, ringTip)
  const pinkyExtended = isFingerExtended(pinkyMCP, pinkyPIP, pinkyTip)

  return [indexExtended, middleExtended, ringExtended, pinkyExtended].filter(Boolean).length
}

function debounce(callback, wait) {
  let timerId
  return (...args) => {
    clearTimeout(timerId)
    timerId = setTimeout(() => {
      callback(...args)
    }, wait)
  }
}

function setProgress(percent) {
  const circle = document.querySelector('.progress-circle')
  const fill = circle.querySelectorAll('.fill')
  const text = circle.querySelector('.inside-circle')

  // Cap percent between 0 and 100
  const cappedPercent = Math.min(Math.max(percent, 0), 100)

  // Update the progress text
  text.innerText = `${cappedPercent}%`

  // Calculate the rotation angle (360deg = 100%)
  const rotation = (cappedPercent / 100) * 360

  // Apply the rotation
  if (rotation <= 180) {
    fill[0].style.transform = `rotate(${rotation}deg)`
    fill[1].style.transform = 'rotate(0deg)' // reset the second fill for values below 50%
  } else {
    fill[0].style.transform = 'rotate(180deg)'
    fill[1].style.transform = `rotate(${rotation - 180}deg)`
  }
}

app()
