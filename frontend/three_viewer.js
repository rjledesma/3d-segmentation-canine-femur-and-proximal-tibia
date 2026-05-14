import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const container = document.getElementById("threeViewer");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x06101f);

const camera = new THREE.PerspectiveCamera(
  45,
  container.clientWidth / container.clientHeight,
  0.1,
  100000
);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
});

renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.domElement.style.display = "block";
renderer.domElement.style.width = "100%";
renderer.domElement.style.height = "100%";
renderer.domElement.style.cursor = "grab";

container.appendChild(renderer.domElement);

// Prevent page scrolling when zooming over the 3D viewer.
container.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
  },
  { passive: false }
);

const controls = new OrbitControls(camera, renderer.domElement);

controls.enableDamping = true;
controls.dampingFactor = 0.08;

controls.enableZoom = true;
controls.zoomSpeed = 2.0;

controls.enableRotate = true;
controls.rotateSpeed = 0.55;

controls.enablePan = true;
controls.panSpeed = 0.85;
controls.screenSpacePanning = true;

controls.autoRotate = false;

// These are updated after model fitting.
controls.minDistance = 0.05;
controls.maxDistance = 20000;

controls.zoomToCursor = true;

renderer.domElement.addEventListener("mousedown", () => {
  renderer.domElement.style.cursor = "grabbing";
});

renderer.domElement.addEventListener("mouseup", () => {
  renderer.domElement.style.cursor = "grab";
});

renderer.domElement.addEventListener("mouseleave", () => {
  renderer.domElement.style.cursor = "grab";
});

scene.add(new THREE.AmbientLight(0xffffff, 0.95));

const light1 = new THREE.DirectionalLight(0x8fdcff, 1.6);
light1.position.set(300, 500, 500);
scene.add(light1);

const light2 = new THREE.DirectionalLight(0xffffff, 0.9);
light2.position.set(-300, -200, 300);
scene.add(light2);

const light3 = new THREE.DirectionalLight(0xffffff, 0.45);
light3.position.set(0, 0, -500);
scene.add(light3);

const modelGroup = new THREE.Group();
scene.add(modelGroup);

let gridGroup = null;

const loader = new STLLoader();

const femurMaterial = new THREE.MeshStandardMaterial({
  color: 0x7effb2,
  roughness: 0.38,
  metalness: 0.06,
  side: THREE.DoubleSide,
});

const tibiaMaterial = new THREE.MeshStandardMaterial({
  color: 0xffd86b,
  roughness: 0.38,
  metalness: 0.06,
  side: THREE.DoubleSide,
});

function loadSTL(path, material, name) {
  return new Promise((resolve, reject) => {
    loader.load(
      path,
      (geometry) => {
        geometry.computeVertexNormals();

        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = name;

        modelGroup.add(mesh);

        console.log(`Loaded ${name}: ${path}`);
        resolve(mesh);
      },
      undefined,
      (error) => {
        console.error(`Failed to load ${name}:`, error);
        reject(error);
      }
    );
  });
}

function removeOldGrid() {
  if (!gridGroup) return;

  scene.remove(gridGroup);

  gridGroup.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  });

  gridGroup = null;
}

function addMedicalGrid() {
  removeOldGrid();

  gridGroup = new THREE.Group();

  const grid = new THREE.GridHelper(420, 20, 0x1d6ea8, 0x123456);
  grid.material.transparent = true;
  grid.material.opacity = 0.32;

  // Background grid, not a floor.
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -120;

  gridGroup.add(grid);
  scene.add(gridGroup);
}

function getBox(object) {
  const box = new THREE.Box3().setFromObject(object);
  return box;
}

function fitCameraToBox(box, zoomMultiplier = 1.65, view = "front") {
  if (!box || box.isEmpty()) return;

  const center = new THREE.Vector3();
  const size = new THREE.Vector3();

  box.getCenter(center);
  box.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * Math.PI / 180;
  const distance = (maxDim / 2) / Math.tan(fov / 2) * zoomMultiplier;

  let direction;

  if (view === "side") {
    direction = new THREE.Vector3(distance, distance * 0.08, distance * 0.15);
  } else if (view === "top") {
    direction = new THREE.Vector3(0, distance, distance * 0.05);
  } else {
    direction = new THREE.Vector3(0, 0, distance);
  }

  camera.position.copy(center).add(direction);

  camera.near = Math.max(distance / 100, 0.1);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);

  controls.minDistance = Math.max(distance * 0.08, 2);
  controls.maxDistance = distance * 6;

  camera.lookAt(center);
  controls.update();
}

function centerAndScaleModel(group) {
  group.position.set(0, 0, 0);
  group.scale.set(1, 1, 1);
  group.rotation.set(0, 0, 0);

  // Make the bones more vertical in the viewer.
  group.rotation.z = Math.PI / 2;

  let box = getBox(group);

  if (box.isEmpty()) {
    console.warn("Model box is empty.");
    return;
  }

  let center = new THREE.Vector3();
  let size = new THREE.Vector3();

  box.getCenter(center);
  box.getSize(size);

  // Center after rotation.
  group.position.sub(center);

  box = getBox(group);
  box.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z);
  const targetSize = 320;
  const scale = targetSize / maxDim;

  group.scale.setScalar(scale);

  // Recenter after scale.
  box = getBox(group);
  box.getCenter(center);
  group.position.sub(center);

  box = getBox(group);
  fitCameraToBox(box, 1.75, "front");

  console.log("Model centered and fitted.");
}

function focusOnObject(object, zoomMultiplier = 1.7) {
  if (!object) {
    fitCameraToBox(getBox(modelGroup), 1.75, "front");
    return;
  }

  const box = getBox(object);
  fitCameraToBox(box, zoomMultiplier, "front");
}

function focusOnKneeRegion() {
  const femur = modelGroup.getObjectByName("Femur");
  const tibia = modelGroup.getObjectByName("Proximal Tibia");

  if (!femur || !tibia) {
    focusOnObject(modelGroup, 1.7);
    return;
  }

  const femurBox = getBox(femur);
  const tibiaBox = getBox(tibia);

  const combinedBox = new THREE.Box3();
  combinedBox.union(femurBox);
  combinedBox.union(tibiaBox);

  if (combinedBox.isEmpty()) {
    focusOnObject(modelGroup, 1.7);
    return;
  }

  const center = new THREE.Vector3();
  const size = new THREE.Vector3();

  combinedBox.getCenter(center);
  combinedBox.getSize(size);

  // Smaller box around the middle where femur and proximal tibia are closest.
  const kneeBox = new THREE.Box3(
    new THREE.Vector3(
      center.x - size.x * 0.28,
      center.y - size.y * 0.28,
      center.z - size.z * 0.28
    ),
    new THREE.Vector3(
      center.x + size.x * 0.28,
      center.y + size.y * 0.28,
      center.z + size.z * 0.28
    )
  );

  fitCameraToBox(kneeBox, 2.15, "front");
}

function setupViewButtons() {
  const buttons = document.querySelectorAll("[data-view]");

  buttons.forEach((oldButton) => {
    const newButton = oldButton.cloneNode(true);
    oldButton.replaceWith(newButton);
  });

  const freshButtons = document.querySelectorAll("[data-view]");

  freshButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const view = button.dataset.view;

      const femur = modelGroup.getObjectByName("Femur");
      const tibia = modelGroup.getObjectByName("Proximal Tibia");

      if (view === "full") {
        focusOnObject(modelGroup, 1.75);
      }

      if (view === "femur") {
        focusOnObject(femur || modelGroup, 1.55);
      }

      if (view === "tibia") {
        focusOnObject(tibia || modelGroup, 1.65);
      }

      if (view === "knee") {
        focusOnKneeRegion();
      }
    });
  });
}

function clearModel() {
  while (modelGroup.children.length > 0) {
    const child = modelGroup.children[0];
    modelGroup.remove(child);

    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  }
}

async function loadInitialModels() {
  try {
    await Promise.all([
      loadSTL("models/femur.stl", femurMaterial, "Femur"),
      loadSTL("models/proximal_tibia.stl", tibiaMaterial, "Proximal Tibia"),
    ]);

    centerAndScaleModel(modelGroup);
    addMedicalGrid();
    setupViewButtons();
  } catch (error) {
    console.error("Could not load initial STL files:", error);
  }
}

window.loadSegmentationSTL = async function (femurUrl, tibiaUrl) {
  clearModel();

  try {
    await Promise.all([
      loadSTL(`http://127.0.0.1:8000${femurUrl}`, femurMaterial, "Femur"),
      loadSTL(`http://127.0.0.1:8000${tibiaUrl}`, tibiaMaterial, "Proximal Tibia"),
    ]);

    centerAndScaleModel(modelGroup);
    addMedicalGrid();
    setupViewButtons();

    console.log("Loaded backend STL outputs.");
  } catch (error) {
    console.error("Failed to load backend STL files:", error);
  }
};

renderer.domElement.addEventListener("dblclick", () => {
  focusOnObject(modelGroup, 1.75);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  const width = container.clientWidth;
  const height = container.clientHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  renderer.setSize(width, height);
});

loadInitialModels();
animate();