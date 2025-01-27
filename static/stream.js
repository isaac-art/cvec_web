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
const moonGeometry = new THREE.SphereGeometry(3, 64, 64);
const textureLoader = new THREE.TextureLoader();

// Load moon textures
const moonTexture = textureLoader.load('/static/textures/moon_texture.jpg');
const moonBumpMap = textureLoader.load('/static/textures/moon_bump.jpg');

const moonMaterial = new THREE.MeshPhongMaterial({
    map: moonTexture,
    bumpMap: moonBumpMap,
    bumpScale: 0.02,
    shininess: 0,
});

const moon = new THREE.Mesh(moonGeometry, moonMaterial);
scene.add(moon);

// Earth setup (for shadow casting)
const earthGeometry = new THREE.SphereGeometry(5, 64, 64);
const earthMaterial = new THREE.MeshPhongMaterial({
    color: 0x000000,
    transparent: true,
    opacity: 0,
});
const earth = new THREE.Mesh(earthGeometry, earthMaterial);
earth.position.set(-8, 0, 0);
scene.add(earth);

// Lighting setup
const ambientLight = new THREE.AmbientLight(0x090909, 0.1);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 1.5);
sunLight.position.set(50, 0, 0);

// Enable shadow casting
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

sunLight.castShadow = true;
earth.castShadow = true;
moon.receiveShadow = true;

// Improve shadow quality
sunLight.shadow.mapSize.width = 2048;
sunLight.shadow.mapSize.height = 2048;
sunLight.shadow.camera.near = 0.1;
sunLight.shadow.camera.far = 100;
sunLight.shadow.camera.left = -10;
sunLight.shadow.camera.right = 10;
sunLight.shadow.camera.top = 10;
sunLight.shadow.camera.bottom = -10;

scene.add(sunLight);

// Adjust camera position to be further back to accommodate larger moon
camera.position.z = 10;

// Add these variables at the top level
let currentTokenStrength = 0;
let targetTokenStrength = 0;
const LERP_SPEED = 0.05;

function getLibrationAngles() {
    const now = Date.now() / 1000;
    
    // Lunar libration parameters (in radians)
    const INCLINATION = 6.687 * Math.PI / 180;    // Moon's orbital inclination
    const TILT = 1.54 * Math.PI / 180;            // Moon's axis tilt
    
    // Periods (in seconds for demo, would normally be in days)
    const NODAL_PERIOD = 60;                      // Simplified from 27.212 days
    const ANOMALISTIC_PERIOD = 55;                // Simplified from 27.554 days
    const EARTH_ROTATION_PERIOD = 10;             // Simplified from 24 hours
    
    // Optical libration in latitude (due to Moon's orbital inclination)
    const latitudeAngle = INCLINATION * Math.sin(2 * Math.PI * now / NODAL_PERIOD);
    
    // Optical libration in longitude (due to Moon's elliptical orbit)
    const longitudeAngle = 7.9 * (Math.PI / 180) * Math.sin(2 * Math.PI * now / ANOMALISTIC_PERIOD);
    
    // Diurnal libration (daily oscillation due to Earth's rotation)
    const diurnalLat = 0.5 * (Math.PI / 180) * Math.sin(2 * Math.PI * now / EARTH_ROTATION_PERIOD);
    const diurnalLong = 0.5 * (Math.PI / 180) * Math.cos(2 * Math.PI * now / EARTH_ROTATION_PERIOD);
    
    // Combine the libration effects
    const totalLatitude = latitudeAngle + diurnalLat;
    const totalLongitude = longitudeAngle + diurnalLong;
    
    // Physical libration (small oscillation of the Moon itself)
    const physicalLibration = 0.02 * Math.PI / 180 * Math.sin(2 * Math.PI * now / ANOMALISTIC_PERIOD);
    
    return {
        latitudeAngle: totalLatitude,
        longitudeAngle: totalLongitude,
        physicalLibration
    };
}

function animate() {
    requestAnimationFrame(animate);

    const { latitudeAngle, longitudeAngle, physicalLibration } = getLibrationAngles();
    
    moon.rotation.x = latitudeAngle;
    moon.rotation.y = longitudeAngle + physicalLibration;
    moon.rotation.z = 0;
    
    currentTokenStrength = THREE.MathUtils.lerp(
        currentTokenStrength,
        targetTokenStrength,
        LERP_SPEED
    );
    
    // Map the strength values to moon phases:
    // -0.6 = new moon (sun behind moon)
    // 0.0 = half moon (waxing)
    // 0.9 = full moon (sun in front)
    
    // First, normalize the range from -0.6 to 0.9 to 0 to 1
    const normalizedStrength = (currentTokenStrength - (-0.6)) / (0.9 - (-0.6));
    
    // Convert to angle where:
    // 0 = sun behind moon (-0.6 strength)
    // π/2 = sun at right side (0.0 strength)
    // π = sun in front (0.9 strength)
    const phase = normalizedStrength * Math.PI;
    
    // Position sun to orbit around the moon
    sunLight.position.x = 50 * Math.sin(phase);  // Right side is positive X
    sunLight.position.z = -50 * Math.cos(phase); // Front is negative Z
    
    // Earth is always opposite the sun
    earth.position.x = -8 * Math.sin(phase);
    earth.position.z = 8 * Math.cos(phase);

    sunLight.lookAt(moon.position);

    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    updateCameraAspect();
});

// Start animation
animate();

// Text stream handling
function strengthToStyle(strength) {
    // Convert strength [-0.5, 0.9] to lightness [10, 255]
    const normalized = (strength - (-0.6)) / (0.9 - (-0.6));
    const lightness = 100 + (normalized * (255 - 120));
    
    // Calculate stroke opacity (inverse of lightness)
    // When lightness is 255, opacity is 0
    // When lightness is 10, opacity is 1
    const strokeOpacity = 1 - (lightness / 255);
    
    // Convert strength to skew angle (-15 to 15 degrees)
    const skewAngle = (strength * 30) - 15;
    
    return {
        color: `rgb(${lightness}, ${lightness}, ${lightness})`,
        transform: `skew(${skewAngle}deg)`,
    };
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
    
    // Update the target strength for moon phase
    targetTokenStrength = data.strength;
    
    // Check if content is empty space and add line break instead
    if (data.content.trim() === ' ' || data.content.trim() === '\n') {
        textContainer.appendChild(document.createElement('br'));
        return;
    }

    // Check content for special words and determine classes
    const content = data.content.toLowerCase();
    const hasLunarTerms = content.includes('moon') || content.includes('lunar');
    const hasAITerms = content.includes('ai') || 
                      content.includes('artificial') || 
                      content.includes('intelligence');
    
    let specialClasses = [];
    if (hasLunarTerms) specialClasses.push('word-moon');
    if (hasAITerms) specialClasses.push('word-ai');
    
    // Create wrapper div for token and its strength
    const wrapper = document.createElement('div');
    wrapper.style.display = 'inline-block';
    wrapper.style.position = 'relative';
    wrapper.style.verticalAlign = 'bottom';
    
    // Create token span
    const span = document.createElement('span');
    span.textContent = data.content;
    span.className = 'token';
    if (specialClasses.length > 0) {
        span.classList.add(...specialClasses);
    }
    
    // Create strength and token_id label
    const infoLabel = document.createElement('div');
    infoLabel.textContent = `${data.strength.toFixed(2)}`; // [${data.token_id}]`;
    infoLabel.style.position = 'absolute';
    infoLabel.style.bottom = '8px';
    infoLabel.style.left = '20px';
    infoLabel.style.fontSize = '8px';
    infoLabel.style.fontFamily = 'monospace';
    infoLabel.style.color = '#666';
    infoLabel.style.fontStyle = 'normal';
    infoLabel.style.lineHeight = '1';
    infoLabel.style.whiteSpace = 'nowrap';
    
    const style = strengthToStyle(data.strength);
    span.style.color = style.color;
    span.style.transform = style.transform;
    span.style.textShadow = style.textShadow;
    
    wrapper.appendChild(span);
    // wrapper.appendChild(infoLabel);
    textContainer.appendChild(wrapper);
    
    if (!userHasScrolled) {
        requestAnimationFrame(() => {
            textContainer.scrollTop = textContainer.scrollHeight;
        });
    }
};

// Update the text container styles to allow overlap
textContainer.style.width = '40%';  // Increase width slightly
textContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.0)'; // More transparent background
textContainer.style.zIndex = '2'; // Ensure text stays on top

// Update the scene container to be full width but behind text
sceneContainer.style.left = '0';  // Start from left edge
sceneContainer.style.width = '100%';  // Take full width
sceneContainer.style.zIndex = '1'; // Behind text

// Update camera aspect ratio to match new container dimensions
function updateCameraAspect() {
    const width = sceneContainer.clientWidth;
    const height = sceneContainer.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Initial aspect ratio update
updateCameraAspect(); 