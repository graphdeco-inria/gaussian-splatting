'use strict'; 

//var socket = io();
var socket = io.connect(null, {port: 5000, rememberTransport: false});
socket.on('connect', function(){
    socket.emit('my_event', {data: 'I\'m connected!'});
});

// Get the canvas and button elements
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
const img1 = new Image()

const drawButton = document.getElementById('drawButton');

// listen for img1
socket.on('img1', function(msg) {
        console.log("Get Image!")
        let arrayBufferView = new Uint8Array(msg['image']);
        console.log(arrayBufferView);

        var blob = new Blob( [ arrayBufferView ], { type: "image/jpeg" } );
        var img1_url = URL.createObjectURL(blob);
        console.log(img1_url);
        img1.onload = function () {
            canvas.height = img1.height;
            canvas.width = img1.width;
            ctx.drawImage(img1, 0, 0);
        }
        img1.src = img1_url
});

// FPS limit
let lastKeyPressedTime = 0;
window.addEventListener("keypress", keyEventHandler, false);
function keyEventHandler(event){
       const currentTime = new Date().getTime();
       if (currentTime - lastKeyPressedTime > 100) { // 100ms = 0.1 second
           lastKeyPressedTime = currentTime;
        socket.emit("key_control", {key: event.key})
        console.log(event.key);
       } else {
          console.log("Too many requests!");
       }
}

// Set initial drawing state
let isDrawing = false;
let startX = 0;
let startY = 0;

// Event listener for the "Connect" button
drawButton.addEventListener('click', () => {
    // Set the drawing state to true
    isDrawing = true;

});

// Event listener for mouse down on the canvas
canvas.addEventListener('mousedown', (e) => {
    if (isDrawing) {
        // Set the starting point for the line
        startX = e.clientX - canvas.getBoundingClientRect().left;
        startY = e.clientY - canvas.getBoundingClientRect().top;
    }
});

// Event listener for mouse up on the canvas
canvas.addEventListener('mouseup', () => {
    // If drawing was in progress, draw the line
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(event.clientX - canvas.getBoundingClientRect().left, event.clientY - canvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});

// Event listener for mouse leave on the canvas
canvas.addEventListener('mouseleave', () => {
    // If drawing was in progress, stop drawing
    if (isDrawing) {
        isDrawing = false;
    }
});
