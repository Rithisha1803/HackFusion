const socket = io();
const localVideo = document.getElementById('localVideo');
const videosContainer = document.getElementById('videos');
const peers = {};
let localStream;
let audioLevels = {};
let participants = {};

// ===== Get Local Media =====
async function initMedia() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localVideo.srcObject = localStream;
        tryStartGaze();

        // ===== Gaze Detection using MediaPipe FaceMesh =====
        let faceMesh;
        let gazeState = { lastDirection: "center", changes: 0, awaySeconds: 0, lastTimestamp: Date.now() };
        let gazeIntervalHandle;
        const gazeEmitDebounce = 300; // ms to debounce same-direction emits

        // create an overlay canvas on top of localVideo
        const overlay = document.createElement('canvas');
        overlay.id = 'gazeOverlay';
        overlay.style.position = 'absolute';
        overlay.style.left = localVideo.offsetLeft + 'px';
        overlay.style.top = localVideo.offsetTop + 'px';
        overlay.style.pointerEvents = 'none';
        overlay.width = localVideo.clientWidth;
        overlay.height = localVideo.clientHeight;
        document.body.appendChild(overlay);
        const octx = overlay.getContext('2d');

        function resizeOverlay() {
            overlay.width = localVideo.clientWidth;
            overlay.height = localVideo.clientHeight;
            overlay.style.left = localVideo.getBoundingClientRect().left + 'px';
            overlay.style.top = localVideo.getBoundingClientRect().top + 'px';
        }
        window.addEventListener('resize', resizeOverlay);

        // mapping helper: normalized landmark to pixel in video coords
        function toVideoCoords(landmark, video) {
            return {
                x: landmark.x * video.videoWidth,
                y: landmark.y * video.videoHeight
            };
        }

        function startGazeDetection() {
            faceMesh = new FaceMesh({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
            });
            faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            faceMesh.onResults(onFaceMeshResults);

            const camera = new Camera(localVideo, {
                onFrame: async () => {
                    await faceMesh.send({ image: localVideo });
                },
                width: 640,
                height: 480
            });
            camera.start();
            resizeOverlay();
        }

        function onFaceMeshResults(results) {
            octx.clearRect(0, 0, overlay.width, overlay.height);

            if (!results.multiFaceLandmarks || !results.multiFaceLandmarks[0]) {
                if (!window._lastFaceSeen) window._lastFaceSeen = Date.now();
                if (Date.now() - window._lastFaceSeen > 1000) handleDirection("away");
                return;
            }
            window._lastFaceSeen = Date.now();

            const landmarks = results.multiFaceLandmarks[0];

            const leftOuter = toVideoCoords(landmarks[33], localVideo);
            const leftInner = toVideoCoords(landmarks[133], localVideo);
            const leftIris = toVideoCoords(landmarks[468], localVideo);
            const rightOuter = toVideoCoords(landmarks[263], localVideo);
            const rightInner = toVideoCoords(landmarks[362], localVideo);
            const rightIris = toVideoCoords(landmarks[473], localVideo);

            drawDot(leftIris, 'rgba(255,0,0,0.8)');
            drawDot(rightIris, 'rgba(255,0,0,0.8)');

            const irisRatio = (inner, outer, iris) => {
                const eyeWidth = Math.hypot(outer.x - inner.x, outer.y - inner.y);
                const irisOffset = Math.hypot(iris.x - inner.x, iris.y - inner.y);
                return irisOffset / eyeWidth;
            };
            const lratio = irisRatio(leftInner, leftOuter, leftIris);
            const rratio = irisRatio(rightInner, rightOuter, rightIris);
            const horizRatio = (lratio + (1 - rratio)) / 2;

            if (!window._horizHistory) window._horizHistory = [];
            window._horizHistory.push(horizRatio);
            if (window._horizHistory.length > 5) window._horizHistory.shift();
            const avgHoriz = window._horizHistory.reduce((a, b) => a + b, 0) / window._horizHistory.length;

            const leyeCenterY = (leftInner.y + leftOuter.y) / 2;
            const reyeCenterY = (rightInner.y + rightOuter.y) / 2;
            const eyeCenterY = (leyeCenterY + reyeCenterY) / 2;
            const irisCenterY = (leftIris.y + rightIris.y) / 2;
            const yDiff = (irisCenterY - eyeCenterY) / (localVideo.videoHeight || overlay.height);

            let direction = "center";
            if (yDiff < -0.05) direction = "top";
            else if (yDiff > 0.07) direction = "bottom";
            else if (avgHoriz < 0.40) direction = "left";
            else if (avgHoriz > 0.60) direction = "right";
            else direction = "center";

            handleDirection(direction);
        }

        function drawDot(pt, color = 'red') {
            const scaleX = overlay.width / localVideo.videoWidth;
            const scaleY = overlay.height / localVideo.videoHeight;
            octx.fillStyle = color;
            octx.beginPath();
            octx.arc(pt.x * scaleX, pt.y * scaleY, 4, 0, 2 * Math.PI);
            octx.fill();
        }

        let lastEmit = 0;
        function handleDirection(direction) {
            const now = Date.now();
            if (direction !== gazeState.lastDirection) {
                if (!window._pendingDirection) {
                    window._pendingDirection = { dir: direction, since: now, count: 1 };
                } else {
                    if (window._pendingDirection.dir === direction) {
                        window._pendingDirection.count += 1;
                    } else {
                        window._pendingDirection = { dir: direction, since: now, count: 1 };
                    }
                }
                if (window._pendingDirection.count >= 2) {
                    gazeState.changes += 1;
                    gazeState.lastDirection = direction;
                    gazeState.lastTimestamp = now;
                    window._pendingDirection = null;
                    if (now - lastEmit > gazeEmitDebounce) {
                        lastEmit = now;
                        socket.emit('gaze-event', {
                            socketId: socket.id,
                            meetingId: meetingId,
                            direction,
                            timestamp: new Date().toISOString()
                        });
                    }
                }
            } else {
                window._pendingDirection = null;
            }
        }

        function tryStartGaze() {
            if (localVideo && localVideo.readyState >= 1) startGazeDetection();
            else localVideo.onloadedmetadata = () => startGazeDetection();
        }

        localVideo.addEventListener('loadeddata', resizeOverlay);

        startAudioDetection(localStream, socket.id);
        participants[socket.id] = 'You';
        updateParticipants();
        socket.emit('join-meeting', { meetingId });
    } catch (err) {
        console.error("Camera access error:", err);
        if (window.showCameraError) window.showCameraError();
    }
}
initMedia();

// ===== Audio Detection =====
function startAudioDetection(stream, id) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 512;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    function detect() {
        analyser.getByteFrequencyData(dataArray);
        let sum = dataArray.reduce((a, b) => a + b, 0);
        audioLevels[id] = sum;
        updateActiveSpeaker();
        requestAnimationFrame(detect);
    }
    detect();
}

function updateActiveSpeaker() {
    const maxId = Object.keys(audioLevels).reduce((a, b) => audioLevels[a] > audioLevels[b] ? a : b);
    document.querySelectorAll('.videoContainer').forEach(vc => vc.classList.remove('activeSpeaker'));
    const active = document.getElementById(`remoteContainer-${maxId}`);
    if (active) active.classList.add('activeSpeaker');
    if (maxId === socket.id) document.getElementById('localContainer').classList.add('activeSpeaker');
    else document.getElementById('localContainer').classList.remove('activeSpeaker');
}

// ===== Remote Video Handling =====
function addRemoteVideo(stream, id) {
    if (document.getElementById(`remoteContainer-${id}`)) return;
    const container = document.createElement("div");
    container.className = "videoContainer";
    container.id = `remoteContainer-${id}`;
    const video = document.createElement("video");
    video.id = `remoteVideo-${id}`;
    video.autoplay = true;
    video.playsInline = true;
    video.srcObject = stream;
    const label = document.createElement("div");
    label.className = "videoLabel";
    label.id = `videoLabel-${id}`;
    label.innerText = participants[id] || "Peer";
    container.appendChild(video);
    container.appendChild(label);
    videosContainer.appendChild(container);
    startAudioDetection(stream, id);
    updateGrid();
}

function removeRemoteVideo(id) {
    const container = document.getElementById(`remoteContainer-${id}`);
    if (container) container.remove();
    delete audioLevels[id];
    delete participants[id];
    updateParticipants();
    updateGrid();
}

function updateGrid() {
    const count = videosContainer.children.length;
    videosContainer.style.gridTemplateColumns = `repeat(auto-fit,minmax(${Math.min(200, 800 / count)}px,1fr))`;
}

// ===== Socket.IO signaling =====
socket.on('new-participant', async ({ socketId }) => {
    if (socketId === socket.id) return;
    participants[socketId] = "Peer";
    updateParticipants();
    const pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
    peers[socketId] = pc;
    localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
    pc.ontrack = e => addRemoteVideo(e.streams[0], socketId);
    pc.onicecandidate = e => { if (e.candidate) socket.emit('signal', { to: socketId, iceCandidate: e.candidate }); };
    if (isHost) {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        socket.emit('signal', { to: socketId, offer });
    }
});

socket.on('signal', async data => {
    const from = data.from;
    if (!peers[from]) {
        const pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
        peers[from] = pc;
        localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
        pc.ontrack = e => addRemoteVideo(e.streams[0], from);
        pc.onicecandidate = e => { if (e.candidate) socket.emit('signal', { to: from, iceCandidate: e.candidate }); };
    }
    const pc = peers[from];
    if (data.offer) {
        await pc.setRemoteDescription(new RTCSessionDescription(data.offer));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        socket.emit('signal', { to: from, answer });
    } else if (data.answer) {
        await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
    } else if (data.iceCandidate) {
        await pc.addIceCandidate(data.iceCandidate);
    }
});

socket.on('disconnect-peer', id => {
    if (peers[id]) { peers[id].close(); delete peers[id]; }
    removeRemoteVideo(id);
});

// ===== Controls =====
document.getElementById('muteBtn').onclick = () => {
    const track = localStream.getAudioTracks()[0];
    const enabled = track.enabled;
    track.enabled = !enabled;
    document.getElementById('muteBtn').innerHTML = enabled ? '<i class="fa-solid fa-microphone-slash"></i>' : '<i class="fa-solid fa-microphone"></i>';
};

// ✅ Fixed video toggle
document.getElementById('videoBtn').onclick = async () => {
    const videoTrack = localStream.getVideoTracks()[0];
    const isEnabled = videoTrack.enabled;

    if (isEnabled) {
        // 🔴 Turn off camera
        videoTrack.enabled = false;
        document.getElementById('videoBtn').innerHTML = '<i class="fa-solid fa-video-slash"></i>';
        for (let peerId in peers) {
            const sender = peers[peerId].getSenders().find(s => s.track && s.track.kind === 'video');
            if (sender) sender.replaceTrack(null);
        }
        localVideo.srcObject = null;
        const black = document.createElement('canvas');
        black.width = 640; black.height = 480;
        localVideo.srcObject = black.captureStream();
    } else {
        // 🟢 Turn camera back on
        const newStream = await navigator.mediaDevices.getUserMedia({ video: true });
        const newTrack = newStream.getVideoTracks()[0];
        localStream.removeTrack(videoTrack);
        localStream.addTrack(newTrack);
        localVideo.srcObject = localStream;
        for (let peerId in peers) {
            const sender = peers[peerId].getSenders().find(s => s.track && s.track.kind === 'video');
            if (sender) sender.replaceTrack(newTrack);
        }
        document.getElementById('videoBtn').innerHTML = '<i class="fa-solid fa-video"></i>';
    }
};

document.getElementById('screenBtn').onclick = async () => {
    try {
        const screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        const screenTrack = screenStream.getVideoTracks()[0];
        for (let peerId in peers) {
            const sender = peers[peerId].getSenders().find(s => s.track.kind === 'video');
            sender.replaceTrack(screenTrack);
        }
        localVideo.srcObject = screenStream;
        screenTrack.onended = () => {
            for (let peerId in peers) {
                const sender = peers[peerId].getSenders().find(s => s.track.kind === 'video');
                sender.replaceTrack(localStream.getVideoTracks()[0]);
            }
            localVideo.srcObject = localStream;
        };
    } catch (e) { console.error("Screen share error:", e); }
};

// ===== Chat =====
const messagesDiv = document.getElementById('messages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
sendBtn.onclick = sendMessage;
chatInput.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });

function sendMessage() {
    const msg = chatInput.value.trim();
    if (msg) {
        socket.emit('chat', { message: msg });
        addMessage(`Me: ${msg}`);
        chatInput.value = '';
    }
}
function addMessage(msg) {
    const div = document.createElement('div');
    div.innerText = `${new Date().toLocaleTimeString()} - ${msg}`;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
socket.on('chat', d => addMessage(`Peer: ${d.message}`));

// ===== Participants =====
function updateParticipants() {
    const list = document.getElementById('participantsList');
    list.innerHTML = '';
    for (let id in participants) {
        const div = document.createElement('div');
        div.className = 'participant';
        div.innerHTML = `<div class="avatar">${participants[id][0]}</div> ${participants[id]}`;
        list.appendChild(div);
    }
}

// ===== Raise Hand =====
document.getElementById('raiseHandBtn').onclick = () => {
    socket.emit('raise-hand', { id: socket.id });
};
socket.on('raise-hand', data => {
    const div = document.createElement('div');
    div.innerText = `${participants[data.id] || data.id} ✋ raised hand!`;
    div.style.color = '#ffd700';
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
});

// ===== Leave Meeting =====
document.getElementById('leaveBtn').onclick = () => {
    location.href = '/clientportal';
};

// ===== Gaze Update UI =====
socket.on('gaze-update', data => {
    const { socketId, direction } = data;
    const label = document.getElementById(`videoLabel-${socketId}`);
    if (label) {
        const peerName = participants[socketId] || 'Peer';
        const emojiMap = { left: '👈', right: '👉', center: '👁️', top: '☝️', bottom: '👇', away: '⚠️' };
        const emoji = emojiMap[direction] || '👁️';
        label.textContent = `${peerName} ${emoji} ${direction}`;
        if (direction === 'center') label.style.backgroundColor = 'rgba(0,255,0,0.7)';
        else if (direction === 'away') label.style.backgroundColor = 'rgba(255,0,0,0.7)';
        else label.style.backgroundColor = 'rgba(255,165,0,0.7)';
    }
});
