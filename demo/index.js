/* eslint-disable object-curly-spacing */
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`
);

const stats = new Stats();
stats.showPanel(0);
document.body.prepend(stats.domElement);

let model;
let ctx;
let videoWidth;
let videoHeight;
let video;
let canvas;
let canvasBreak = document.getElementById('breakout');
let ctxBreak = canvasBreak.getContext('2d');
let ballRadius = 10;
let x = canvasBreak.width / 2;
let y = canvasBreak.height - 30;
let dx = 2;
let dy = -2;
let paddleHeight = 10;
let paddleWidth = 75;
let paddleX = (canvasBreak.width - paddleWidth) / 2;
let rightPressed = false;
let leftPressed = false;
let brickRowCount = 7;
let brickColumnCount = 5;
let brickWidth = 75;
let brickHeight = 20;
let brickPadding = 10;
let brickOffsetTop = 30;
let brickOffsetLeft = 30;
let score = 0;
let lives = 3;

let bricks = [];
for (let c = 0; c < brickColumnCount; c++) {
  bricks[c] = [];
  for (let r = 0; r < brickRowCount; r++) {
    bricks[c][r] = { x: 0, y: 0, status: 1 };
  }
}

function collisionDetection() {
  for (let c = 0; c < brickColumnCount; c++) {
    for (let r = 0; r < brickRowCount; r++) {
      let b = bricks[c][r];
      if (b.status == 1) {
        if (
          x > b.x &&
          x < b.x + brickWidth &&
          y > b.y &&
          y < b.y + brickHeight
        ) {
          dy = -dy;
          b.status = 0;
          score++;
          if (score == brickRowCount * brickColumnCount) {
            alert('YOU WIN, CONGRATS!');
            document.location.reload();
          }
        }
      }
    }
  }
}

function drawBall() {
  ctxBreak.beginPath();
  ctxBreak.arc(x, y, ballRadius, 0, Math.PI * 2);
  ctxBreak.fillStyle = '#0095DD';
  ctxBreak.fill();
  ctxBreak.closePath();
}
function drawPaddle() {
  ctxBreak.beginPath();
  ctxBreak.rect(
    paddleX,
    canvasBreak.height - paddleHeight,
    paddleWidth,
    paddleHeight
  );
  ctxBreak.fillStyle = '#0095DD';
  ctxBreak.fill();
  ctxBreak.closePath();
}
function drawBricks() {
  for (let c = 0; c < brickColumnCount; c++) {
    for (let r = 0; r < brickRowCount; r++) {
      if (bricks[c][r].status == 1) {
        let brickX = r * (brickWidth + brickPadding) + brickOffsetLeft;
        let brickY = c * (brickHeight + brickPadding) + brickOffsetTop;
        bricks[c][r].x = brickX;
        bricks[c][r].y = brickY;
        ctxBreak.beginPath();
        ctxBreak.rect(brickX, brickY, brickWidth, brickHeight);
        ctxBreak.fillStyle = '#0095DD';
        ctxBreak.fill();
        ctxBreak.closePath();
      }
    }
  }
}
function drawScore() {
  ctxBreak.font = '16px Arial';
  ctxBreak.fillStyle = '#0095DD';
  ctxBreak.fillText('Score: ' + score, 8, 20);
}
function drawLives() {
  ctxBreak.font = '16px Arial';
  ctxBreak.fillStyle = '#0095DD';
  ctxBreak.fillText('Lives: ' + lives, canvasBreak.width - 65, 20);
}

function draw() {
  ctxBreak.clearRect(0, 0, canvasBreak.width, canvasBreak.height);
  drawBricks();
  drawBall();
  drawPaddle();
  drawScore();
  drawLives();
  collisionDetection();

  if (x + dx > canvasBreak.width - ballRadius || x + dx < ballRadius) {
    dx = -dx;
  }
  if (y + dy < ballRadius) {
    dy = -dy;
  } else if (y + dy > canvasBreak.height - ballRadius) {
    if (x > paddleX && x < paddleX + paddleWidth) {
      dy = -dy;
    } else {
      lives--;
      if (!lives) {
        alert('GAME OVER');
        document.location.reload();
      } else {
        x = canvasBreak.width / 2;
        y = canvasBreak.height - 30;
        dx = 2;
        dy = -2;
        paddleX = (canvasBreak.width - paddleWidth) / 2;
      }
    }
  }

  if (rightPressed && paddleX < canvasBreak.width - paddleWidth) {
    paddleX += 7;
  } else if (leftPressed && paddleX > 0) {
    paddleX -= 7;
  }

  x += dx;
  y += dy;
  requestAnimationFrame(draw);
}

draw();

const state = {
  backend: 'wasm',
};

const gui = new dat.GUI();
gui
  .add(state, 'backend', ['wasm', 'webgl', 'cpu'])
  .onChange(async (backend) => {
    await tf.setBackend(backend);
  });

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  stats.begin();

  const returnTensors = false;
  const flipHorizontal = true;
  const annotateBoxes = true;
  const predictions = await model.estimateFaces(
    video,
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
        if (annotateBoxes) {
          predictions[i].landmarks = predictions[i].landmarks.arraySync();
        }
      }

      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];
      // ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
      ctx.fillStyle = 'rgb(224, 247, 250)';
      ctx.fillRect(start[0], start[1], size[0], size[1]);

      if (annotateBoxes) {
        const landmarks = predictions[i].landmarks;

        ctx.fillStyle = 'blue';
        const x = landmarks[2][0];
        const y = landmarks[2][1];
        ctx.fillRect(x, y, 5, 5);
        paddleX = x;
      }
    }
  }

  stats.end();

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';

  model = await blazeface.load();

  renderPrediction();
};

setupPage();
