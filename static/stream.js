import * as THREE from 'three';

const textContainer = document.getElementById('text-container');
const sceneContainer = document.getElementById('scene-container');

// Three.js setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
sceneContainer.appendChild(renderer.domElement);

// Moon setup
const moonGeometry = new THREE.SphereGeometry(2, 32, 32);
const textureLoader = new THREE.TextureLoader();

// Load moon textures
const moonTexture = textureLoader.load('/static/textures/moon_texture.jpg');
const moonBumpMap = textureLoader.load('/static/textures/moon_bump.jpg');

const moonMaterial = new THREE.MeshPhongMaterial({
    map: moonTexture,
    bumpMap: moonBumpMap,
    bumpScale: 0.02,
});

const moon = new THREE.Mesh(moonGeometry, moonMaterial);
scene.add(moon);

// Lighting
const ambientLight = new THREE.AmbientLight(0x202020);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 1);
sunLight.position.set(5, 0, 5);
scene.add(sunLight);

// Camera position
camera.position.z = 6;

// Animation
function getMoonPhase() {
    const now = Date.now() / 1000;
    const period = 60; // 60 second cycle
    return (now % period) / period * Math.PI * 2;
}

function getLibrationAngles() {
    const now = Date.now() / 1000;
    // Simulate monthly libration in latitude (±6.7°)
    const latitudeAngle = Math.sin(now * 0.1) * (6.7 * Math.PI / 180);
    // Simulate monthly libration in longitude (±7.9°)
    const longitudeAngle = Math.sin(now * 0.08) * (7.9 * Math.PI / 180);
    return { latitudeAngle, longitudeAngle };
}

function animate() {
    requestAnimationFrame(animate);

    // Update moon libration
    const { latitudeAngle, longitudeAngle } = getLibrationAngles();
    moon.rotation.x = latitudeAngle;
    moon.rotation.y = longitudeAngle;

    // Update sun position for moon phases
    const phase = getMoonPhase();
    sunLight.position.x = Math.cos(phase) * 5;
    sunLight.position.z = Math.sin(phase) * 5;

    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    
    renderer.setSize(width, height);
});

// Start animation
animate();

// Text stream handling
function strengthToColor(strength) {
    // Convert strength [-0.5, 0.9] to hue [120, 0]
    const normalized = (strength - (-0.5)) / (0.9 - (-0.5));
    const hue = (1 - normalized) * 120;
    return `hsl(${hue}, 80%, 70%)`; // Increased lightness for better visibility
}

// Track if user has manually scrolled up
let userHasScrolled = false;
textContainer.addEventListener('scroll', () => {
    const isAtBottom = textContainer.scrollHeight - textContainer.scrollTop <= textContainer.clientHeight + 50;
    userHasScrolled = !isAtBottom;
});

const evtSource = new EventSource("/stream");
evtSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Add new token with color based on strength
    const span = document.createElement('span');
    span.textContent = data.content;
    span.className = 'token';
    span.style.color = strengthToColor(data.strength);
    textContainer.appendChild(span);
    
    // Auto-scroll only if user hasn't scrolled up
    if (!userHasScrolled) {
        requestAnimationFrame(() => {
            textContainer.scrollTop = textContainer.scrollHeight;
        });
    }
}; 