import * as THREE from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const el = document.getElementById("viewer");
const status = document.getElementById("viewer-status");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
el.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, 0, 3);
const controls = new OrbitControls(camera, renderer.domElement);

function resize() {
  renderer.setSize(el.clientWidth, el.clientHeight);
  camera.aspect = el.clientWidth / el.clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

new PLYLoader().load(
  "assets/pointcloud.ply",
  (geometry) => {
    // Open3D coordinates are in metres and tiny; centre and scale to a unit box.
    // Open3D's y axis points down, so flip it to show the knee upright.
    geometry.center();
    geometry.computeBoundingBox();
    const size = geometry.boundingBox.getSize(new THREE.Vector3());
    const s = 2 / Math.max(size.x, size.y, size.z);
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ size: 0.015, vertexColors: true })
    );
    points.scale.set(s, -s, s);
    scene.add(points);
    status.textContent = "Drag to rotate, scroll to zoom, right-drag to pan.";
  },
  undefined,
  () => { status.textContent = "Could not load the point cloud."; }
);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
