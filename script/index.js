// console.log("1234566")

// UI BEGIN
const video = document.getElementById('video');
const videoControls = document.getElementById('video-controls');
const videoContainer = document.getElementById('video-container');
const classSelectMenu = document.getElementById('class-select-button');

const videoWorks = !!document.createElement('video').canPlayType;

let selected_element = [];

// console.log("enter in index.js")
if (videoWorks) {
    console.log("test for videoWorks")
    video.controls = false;
    videoControls.classList.remove('hidden');
}

// Play/Pause BEGIN
const playButton = document.getElementById('play');
function togglePlay() {
    console.log("toggle Play")
    if (video.paused || video.ended) {
        video.play();
    }
    else {
        video.pause();
    }
}
playButton.addEventListener('click', togglePlay);

const playbackIcons = document.querySelectorAll('.playback-icons use');
function updatePlayButton() {
    playbackIcons.forEach(icon => icon.classList.toggle('hidden'));
    if (video.paused) {
        playButton.setAttribute('data-title', 'Play')
    }
    else {
        playButton.setAttribute('data-title', 'Pause')
    }
}
video.addEventListener('play', updatePlayButton);
video.addEventListener('pause', updatePlayButton);
// Play/Pause END

// Duration/Time elapsed BEGIN
const timeElapsed = document.getElementById('time-elapsed');
const duration = document.getElementById('duration');
function formatTime(timeInSeconds) {
    try {
        const result = new Date(timeInSeconds * 1000).toISOString().substr(11, 8);
        return {
            minutes: result.substr(3, 2),
            seconds: result.substr(6, 2),
        };
    }
    catch (e) {
        console.log('wrong time format');
        return {
            minutes: 'nan',
            seconds: 'nan',
        };
    }
};
// Duration/Time elapsed END

// Progress bar BEGIN
const progressBar = document.getElementById('progress-bar');
const seek = document.getElementById('seek');
function updateVideoInfo() {
    const videoDuration = Math.round(video.duration);
    seek.setAttribute('max', videoDuration);
    progressBar.setAttribute('max', videoDuration);
    const time = formatTime(videoDuration);
    duration.innerText = `${time.minutes}:${time.seconds}`;
    duration.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`);
    video.playbackRate = 0.88;
}
video.addEventListener('loadedmetadata', updateVideoInfo);
// Progress bar END

// Update function BEGIN
function updateTimeElapsed() {
    const time = formatTime(Math.round(video.currentTime));
    timeElapsed.innerText = `${time.minutes}:${time.seconds}`;
    timeElapsed.setAttribute('datetime', `${time.minutes}m ${time.seconds}s`)
}

function updateProgress() {
    seek.value = Math.round(video.currentTime);
    progressBar.value = Math.round(video.currentTime);
}

function updateEverything() {
    updateVideoInfo();
    updateTimeElapsed();
    updateProgress();
}

setInterval(updateEverything, 500);
// Update function END

const seekTooltip = document.getElementById('seek-tooltip');
function updateSeekTooltip(event) {
    const skipTo = Math.round((event.offsetX / event.target.clientWidth) * parseInt(event.target.getAttribute('max'), 10));
    seek.setAttribute('data-seek', skipTo)
    const t = formatTime(skipTo);
    seekTooltip.textContent = `${t.minutes}:${t.seconds}`;
    const rect = video.getBoundingClientRect();
    seekTooltip.style.left = `${event.pageX - rect.left}px`;
}
seek.addEventListener('mousemove', updateSeekTooltip);

function skipAhead(event) {
    const skipTo = event.target.dataset.seek ? event.target.dataset.seek : event.target.value;
    video.currentTime = skipTo;
    progressBar.value = skipTo;
    seek.value = skipTo;
}
seek.addEventListener('input', skipAhead);

// Volume control BEGIN
const volumeButton = document.getElementById('volume-button');
const volumeIcons = document.querySelectorAll('.volume-button use');
const volumeMute = document.querySelector('use[href="#volume-mute"]');
const volumeLow = document.querySelector('use[href="#volume-low"]');
const volumeHigh = document.querySelector('use[href="#volume-high"]');
const volume = document.getElementById('volume');

function updateVolume() {
    if (video.muted) {
        video.muted = false;
    }
    video.volume = volume.value;
}
// volume.addEventListener('input', updateVolume);

function updateVolumeIcon() {
    volumeIcons.forEach(icon => {
        icon.classList.add('hidden');
    });

    volumeButton.setAttribute('data-title', 'Mute')

    if (video.muted || video.volume === 0) {
        volumeMute.classList.remove('hidden');
        volumeButton.setAttribute('data-title', 'Unmute')
    }
    else if (video.volume > 0 && video.volume <= 0.5) {
        volumeLow.classList.remove('hidden');
    }
    else {
        volumeHigh.classList.remove('hidden');
    }
}
video.addEventListener('volumechange', updateVolumeIcon);

function toggleMute() {
    video.muted = !video.muted;
    if (video.muted) {
        volume.setAttribute('data-volume', volume.value);
        volume.value = 0;
    }
    else {
        volume.value = volume.dataset.volume;
    }
}
// volumeButton.addEventListener('click', toggleMute);
// Volume control END
// UI END

var server_ip, flask_port;
$.getJSON('config.json', function (json) {
    server_ip = json.server_ip;
    flask_port = json.flask_port;
});


// Shutdown START
const shutdownButton = document.getElementById('shutdown-button');
function shutdown() {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', `http://${server_ip}:${flask_port}/shutdown`, true);
    xmlHttp.send(null);
}
// shutdownButton.addEventListener('click', shutdown);

// Listen
function getNowObjectCls(){
    console.log("click class-select-menu");
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', `http://${server_ip}:${flask_port}/get_cls`, true);
    xmlHttp.send(null);
    xmlHttp.onload = function () {
        var buttonGroup = document.getElementById('class-select-button-group');
        var buttonGroupSelected = document.getElementById('class-select-button-group-selected');
        buttonGroup.replaceChildren()
        JSON.parse(xmlHttp.responseText).target.forEach( function (cls){
            if (!selected_element.includes(cls)){
                var node = document.createElement('button');
                node.className = `${cls}`
                node.innerHTML = `${cls}`;
                node.onclick = selectTrackObject;
                node.style.backgroundColor = "#4CAF50";
                buttonGroup.appendChild(node);
            }
        });

        // var today = new Date();
        // node.innerHTML = `Detects #${JSON.parse(xmlHttp.responseText).target} @ ${today.timeNow()}`;
        // console.log(JSON.parse(xmlHttp.responseText).target)
        // node.setAttribute('class', 'elem-cls')
        // logBox.insertBefore(node, logBox.firstChild);
    }
    // event.preventDefault();
}
setInterval(getNowObjectCls, 200)


function selectTrackObject(event){
    console.log("click object from buttonGroup list");
    const select_button_value = event.target.value;
    var xmlHttp = new XMLHttpRequest();
   xmlHttp.open(
        'GET',
        `http://${server_ip}:${flask_port}/control_cls?select_cls=` + select_button_value + ',' +  "act=select",
        true
    );
    xmlHttp.send(null);
    xmlHttp.onload = function () {
        document.getElementById(select_button_value).remove();

        var buttonGroupSelected = document.getElementById('class-select-button-group-selected');
        var node = document.createElement('button');
        node.className = `${select_button_value}`
        node.innerHTML = `${select_button_value}`;
        node.onclick = deselectTrackObject;
        node.style.backgroundColor = "#4CAF50";
        buttonGroupSelected.appendChild(node);

        selected_element.push(select_button_value)
    }


}

function deselectTrackObject(event){
    console.log("click object from selected buttonGroup list");
    const select_button_value = event.target.value;
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open(
        'GET',
        `http://${server_ip}:${flask_port}/control_cls?select_cls=` + select_button_value + ',' +  "act=deselect",
        true
    );
    xmlHttp.send(null);
    xmlHttp.onload = function () {
        document.getElementById(select_button_value).remove();
        selected_element.remove(select_button_value);
    }
}




// Shutdown END

// Deselect START
// const deselectButton = document.getElementById('deselect-button');
// function deselect() {
//     var xmlHttp = new XMLHttpRequest();
//     xmlHttp.open('GET', `http://${server_ip}:${flask_port}/data?coor=deselect`, true);
//     xmlHttp.send(null);
//     xmlHttp.onload = function () {
//         var logBox = document.getElementById('log-container');
//         var node = document.createElement('div');
//         var today = new Date();
//         node.innerHTML = `Deselect @ ${today.timeNow()}`;
//         node.setAttribute('class', 'log-elem')
//         logBox.insertBefore(node, logBox.firstChild);
//     }
// }
// deselectButton.addEventListener('click', deselect);
// Deselect END

// Click event START
// function getClickCoordinate(event) {
//     var x = event.clientX + window.pageXOffset - videoContainer.offsetLeft;
//     var y = event.clientY + window.pageYOffset - videoContainer.offsetTop;
//     x = x < 0 ? 0 : x;
//     y = y < 0 ? 0 : y;

//     var xmlHttp = new XMLHttpRequest();
//     xmlHttp.open(
//         'GET',
//         `http://${server_ip}:${flask_port}/data?coor=` + x + ',' + y + ',' + video.offsetHeight + ',' + video.offsetWidth,
//         true
//     );
//     xmlHttp.send(null);
//     xmlHttp.onload = function () {
//         var logBox = document.getElementById('log-container');
//         var node = document.createElement('div');
//         var today = new Date();
//         node.innerHTML = `Select #${JSON.parse(xmlHttp.responseText).target} @ ${today.timeNow()}`;
//         node.setAttribute('class', 'log-elem')
//         logBox.insertBefore(node, logBox.firstChild);
//     }

//     event.preventDefault();
// }
// video.addEventListener('click', getClickCoordinate);
// Click event END

Date.prototype.timeNow = function () {
    return ((this.getHours() < 10) ? '0' : '') + this.getHours() + ':' + ((this.getMinutes() < 10) ? '0' : '') + this.getMinutes() + ':' + ((this.getSeconds() < 10) ? '0' : '') + this.getSeconds();
}





