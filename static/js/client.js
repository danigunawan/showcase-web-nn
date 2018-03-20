
var video = null;
var canvas = null;
var ctx = null;
var smallCanvas = null;
var scctx = null;

var paused=false

var imageSent=false;
$(function(){
    video = document.getElementById("videoElement");
    smallCanvas = document.getElementById("smallCanvas");
    smallCanvas.width=config["IMAGE"]["size_x"]
    smallCanvas.height=config["IMAGE"]["size_y"]
    scctx = smallCanvas.getContext("2d");

    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;

    if (navigator.getUserMedia) {
        navigator.getUserMedia({ video: true }, handleVideo, videoError);
    }
    
    setInterval(mainLoop,5);
    var imgStream=new EventSource('/stream/'+cid+"/stream")

    imgStream.onmessage = function (e) {
	imagesLastSec++
	bufferCurrent--
	$("#buff").text(bufferCurrent)
	image=document.getElementById("output");
	image.src = 'data:image/jpeg;base64,' + e.data;
    };
    setInterval(fpsCounter, 1000);
})
bufferMax=6
bufferCurrent=0
imageSent=false
function mainLoop() {
    if (paused) {
	return
    }
    //if (!imageSent){
	if (bufferCurrent < config["STREAM"]["buffer_size"]) {
	    requestImage()
	}
    //}
}

function fpsCounter() {
    $("#fps").text(imagesLastSec)
    imagesLastSec=0
}

var imagesLastSec=0
function requestImage() {

    var a = performance.now();
    bufferCurrent++
    $("#buff").text(bufferCurrent)
    imageSent=true
    scctx.drawImage(video, 0, 0, config["IMAGE"]["size_x"],  config["IMAGE"]["size_y"]);


    data= smallCanvas.toDataURL("image/jpeg", config["IMAGE"]["c2s_jpeg"]).slice(23)
    var b = performance.now();
    
    var xhr = new XMLHttpRequest;
    xhr.open("POST", "/stream/"+cid+"/push", false);
    xhr.onload = function(e) {
	imageSent=false
    };

    //xhr.send(imageData.data);
    xhr.send(data);


}

function handleVideo(stream) {
    video.src = window.URL.createObjectURL(stream);
}
function videoError(e) {
    alert("error")
    console.log(e)
}
var video = document.getElementById("videoElement");
