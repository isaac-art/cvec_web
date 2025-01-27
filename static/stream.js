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
const moonGeometry = new THREE.SphereGeometry(2, 64, 64);
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

// Camera position adjusted
camera.position.z = 8;

// Updated moon phase calculation
function getMoonPhase() {
    const now = Date.now() / 1000;
    const period = 60; // 60 second cycle
    return (now % period) / period * Math.PI * 2;
}

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

    // Get libration angles
    const { latitudeAngle, longitudeAngle, physicalLibration } = getLibrationAngles();
    
    // Apply libration to moon
    moon.rotation.x = latitudeAngle;                    // Tilt up/down
    moon.rotation.y = longitudeAngle + physicalLibration; // Side to side wobble
    
    // The moon keeps the same face toward Earth (synchronous rotation)
    // We only see the wobble due to libration
    moon.rotation.z = 0; // Moon doesn't spin on its axis relative to Earth
    
    // Update sun and earth positions for realistic moon phases
    const phase = getMoonPhase();
    
    // Move the sun in a larger circle to create sharper shadows
    sunLight.position.x = Math.cos(phase) * 50;
    sunLight.position.z = Math.sin(phase) * 50;
    
    // Update earth position to properly cast shadow
    earth.position.x = Math.cos(phase + Math.PI) * 8;
    earth.position.z = Math.sin(phase + Math.PI) * 8;

    // Keep the sun light pointing at the moon
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
    wrapper.appendChild(infoLabel);
    textContainer.appendChild(wrapper);
    
    if (!userHasScrolled) {
        requestAnimationFrame(() => {
            textContainer.scrollTop = textContainer.scrollHeight;
        });
    }
};

// Update the text container styles
textContainer.style.width = '33%';  // Set to one-third of screen width
textContainer.style.left = '0';
textContainer.style.height = '100vh';  // Full height
textContainer.style.maxHeight = '100vh';  // Override previous max-height

// Update the scene container to take remaining space
sceneContainer.style.left = '33%';  // Start after text container
sceneContainer.style.width = '67%';  // Take remaining width

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