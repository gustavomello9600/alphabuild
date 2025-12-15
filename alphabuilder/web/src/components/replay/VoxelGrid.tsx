import React, { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { CHANNEL, getSupportColor } from './types';
import type { ViewMode } from './types';

// Interface for visual data required by the grid
export interface VoxelVisuals {
    opaqueMatrix: Float32Array;
    opaqueColor: Float32Array;
    overlayMatrix: Float32Array;
    overlayColor: Float32Array;
    mctsMatrix?: Float32Array;
    mctsColor?: Float32Array;
}

interface VoxelGridProps {
    step: {
        visuals?: VoxelVisuals;
        tensor: {
            shape: number[];
            data: Float32Array | number[];
        };
        displacement_map?: Float32Array | null;
        selected_actions?: any[];
        max_displacement?: number;
    } | null;
    nextStep?: {
        tensor: {
            shape: number[];
            data: Float32Array | number[];
        };
    } | null;
    viewMode: ViewMode;
}

export const VoxelGridMCTS: React.FC<VoxelGridProps> = ({ step, nextStep, viewMode }) => {
    const opaqueRef = useRef<THREE.InstancedMesh>(null);
    const overlayRef = useRef<THREE.InstancedMesh>(null);
    const wireframeRef = useRef<THREE.InstancedMesh>(null);
    const opacityAttrRef = useRef<THREE.InstancedBufferAttribute | null>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);

    // Initial Shader setup for opacity
    useEffect(() => {
        if (!overlayRef.current) return;
        const material = overlayRef.current.material as THREE.MeshStandardMaterial;
        material.onBeforeCompile = (shader) => {
            shader.vertexShader = `
                attribute float instanceOpacity;
                varying float vInstanceOpacity;
                ${shader.vertexShader}
            `.replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>
                vInstanceOpacity = instanceOpacity;`
            );
            shader.fragmentShader = `
                varying float vInstanceOpacity;
                ${shader.fragmentShader}
            `.replace(
                '#include <dithering_fragment>',
                `#include <dithering_fragment>
                gl_FragColor.a *= vInstanceOpacity;`
            );
        };
        material.needsUpdate = true;
    }, []);

    // Update Instances when step changes
    useEffect(() => {
        if (!opaqueRef.current || !overlayRef.current || !step) return;

        // Reset counts initially
        opaqueRef.current.count = 0;
        overlayRef.current.count = 0;
        if (wireframeRef.current) wireframeRef.current.count = 0;

        // Custom Decision Mode (Diff View)
        if (viewMode === 'decision') {
            const [, D, H, W] = step.tensor.shape;

            // Ensure we have buffers
            if (!opaqueRef.current.instanceColor || opaqueRef.current.instanceColor.count !== 20000) {
                opaqueRef.current.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(20000 * 3), 3);
            }
            const mat = opaqueRef.current.instanceMatrix.array as Float32Array;
            const col = opaqueRef.current.instanceColor.array as Float32Array;
            let idx = 0;

            const curDensity = step.tensor.data;
            const nextDensity = nextStep ? nextStep.tensor.data : null;

            // Iterate all voxels
            for (let d = 0; d < D; d++) {
                for (let h = 0; h < H; h++) {
                    for (let w = 0; w < W; w++) {
                        const i = d * (H * W) + h * W + w;
                        const isCur = curDensity[i] > 0.5;
                        const isNext = nextDensity ? nextDensity[i] > 0.5 : isCur;

                        if (isCur || isNext) {
                            // Limit to max instance count
                            if (idx >= 20000) break;

                            const x = d - D / 2;
                            const y = h + 0.5;
                            const z = w - W / 2;

                            dummy.position.set(x, y, z);
                            dummy.updateMatrix();
                            dummy.matrix.toArray(mat, idx * 16);

                            // Color Logic
                            if (isCur && !isNext) {
                                // Removed -> Red
                                col[idx * 3] = 1.0; col[idx * 3 + 1] = 0.0; col[idx * 3 + 2] = 0.33;
                            } else if (!isCur && isNext) {
                                // Added -> Green
                                col[idx * 3] = 0.0; col[idx * 3 + 1] = 1.0; col[idx * 3 + 2] = 0.61;
                            } else {
                                // Unchanged -> Standard Gray (Same as Structure View)
                                const density = curDensity[i];
                                const g = Math.max(0.2, density) * 0.95;
                                col[idx * 3] = g; col[idx * 3 + 1] = g; col[idx * 3 + 2] = g;
                            }
                            idx++;
                        }
                    }
                }
            }
            opaqueRef.current.count = idx;
            opaqueRef.current.instanceMatrix.needsUpdate = true;
            opaqueRef.current.instanceColor.needsUpdate = true;
            return;
        }

        // Standard Modes
        if (step.visuals) {
            const { opaqueMatrix, opaqueColor, overlayMatrix, overlayColor, mctsMatrix, mctsColor } = step.visuals;

            // 1. OPAQUE
            const opaqueCount = opaqueMatrix.length / 16;
            opaqueRef.current.count = opaqueCount;
            if (opaqueCount > 0) {
                if (!opaqueRef.current.instanceColor || opaqueRef.current.instanceColor.count !== 20000) {
                    opaqueRef.current.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(20000 * 3), 3);
                }

                // Set Matrices
                opaqueRef.current.instanceMatrix.array.set(opaqueMatrix);
                opaqueRef.current.instanceMatrix.needsUpdate = true;

                // Set Colors (Standard or Deformation)
                if ((viewMode === 'deformation' || viewMode === 'strain') && step.displacement_map) {
                    const [, D, H, W] = step.tensor.shape;
                    const colors = opaqueRef.current.instanceColor.array as Float32Array;
                    const dispMap = step.displacement_map;

                    // FIRST PASS: Collect values for percentile calculation
                    const values: number[] = [];

                    const getDisp = (d: number, h: number, w: number): number => {
                        if (d < 0 || d >= D || h < 0 || h >= H || w < 0 || w >= W) return 0;
                        return dispMap[d * (H * W) + h * W + w] || 0;
                    };

                    for (let i = 0; i < opaqueCount; i++) {
                        const offset = i * 16;
                        const xVal = opaqueMatrix[offset + 12];
                        const yVal = opaqueMatrix[offset + 13];
                        const zVal = opaqueMatrix[offset + 14];

                        const d = Math.round(xVal + D / 2);
                        const h = Math.round(yVal - 0.5);
                        const w = Math.round(zVal + W / 2);

                        if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                            let val = 0;
                            if (viewMode === 'deformation') {
                                const idx = d * (H * W) + h * W + w;
                                val = Math.abs(dispMap[idx]);
                            } else {
                                const dDispDx = (getDisp(d + 1, h, w) - getDisp(d - 1, h, w)) / 2;
                                const dDispDy = (getDisp(d, h + 1, w) - getDisp(d, h - 1, w)) / 2;
                                const dDispDz = (getDisp(d, h, w + 1) - getDisp(d, h, w - 1)) / 2;
                                val = Math.sqrt(dDispDx * dDispDx + dDispDy * dDispDy + dDispDz * dDispDz);
                            }
                            values.push(val);
                        }
                    }

                    // Calculate 99th percentile to ignore outliers
                    let maxMetric = 1.0;
                    if (values.length > 0) {
                        values.sort((a, b) => a - b);
                        const p99 = Math.floor(values.length * 0.99);
                        maxMetric = values[Math.min(p99, values.length - 1)];
                    }
                    if (maxMetric < 1e-6) maxMetric = 1.0;

                    // SECOND PASS: Apply colors
                    for (let i = 0; i < opaqueCount; i++) {
                        const offset = i * 16;
                        const xVal = opaqueMatrix[offset + 12];
                        const yVal = opaqueMatrix[offset + 13];
                        const zVal = opaqueMatrix[offset + 14];

                        const d = Math.round(xVal + D / 2);
                        const h = Math.round(yVal - 0.5);
                        const w = Math.round(zVal + W / 2);

                        let val = 0;

                        if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                            if (viewMode === 'deformation') {
                                const idx = d * (H * W) + h * W + w;
                                val = Math.abs(dispMap[idx]);
                            } else {
                                const dDispDx = (getDisp(d + 1, h, w) - getDisp(d - 1, h, w)) / 2;
                                const dDispDy = (getDisp(d, h + 1, w) - getDisp(d, h - 1, w)) / 2;
                                const dDispDz = (getDisp(d, h, w + 1) - getDisp(d, h, w - 1)) / 2;
                                val = Math.sqrt(dDispDx * dDispDx + dDispDy * dDispDy + dDispDz * dDispDz);
                            }
                        }

                        // Blue (0.66) -> Red (0.0)
                        const norm = Math.min(1.0, val / maxMetric);
                        const hue = (1.0 - norm) * 0.66;
                        const color = new THREE.Color().setHSL(hue, 1.0, 0.5);

                        colors[i * 3] = color.r;
                        colors[i * 3 + 1] = color.g;
                        colors[i * 3 + 2] = color.b;
                    }
                } else {
                    opaqueRef.current.instanceColor.array.set(opaqueColor);
                }
                opaqueRef.current.instanceColor.needsUpdate = true;
            } else {
                opaqueRef.current.count = 0;
            }

            // 2. OVERLAY (Policy / MCTS)
            const showOverlay = viewMode === 'policy' || viewMode === 'combined';
            if (showOverlay) {
                // POLICY (Worker)
                const overlayCount = overlayMatrix.length / 16;
                overlayRef.current.count = overlayCount;
                if (overlayCount > 0) {
                    overlayRef.current.instanceMatrix.array.set(overlayMatrix);
                    overlayRef.current.instanceMatrix.needsUpdate = true;

                    if (!overlayRef.current.instanceColor || overlayRef.current.instanceColor.count !== 20000) {
                        overlayRef.current.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(20000 * 3), 3);
                    }
                    if (!opacityAttrRef.current) {
                        const arr = new Float32Array(20000).fill(1.0);
                        opacityAttrRef.current = new THREE.InstancedBufferAttribute(arr, 1);
                        overlayRef.current.geometry.setAttribute('instanceOpacity', opacityAttrRef.current);
                    }

                    const rgb = overlayRef.current.instanceColor.array as Float32Array;
                    const alpha = opacityAttrRef.current.array as Float32Array;

                    for (let i = 0; i < overlayCount; i++) {
                        rgb[i * 3] = overlayColor[i * 4];
                        rgb[i * 3 + 1] = overlayColor[i * 4 + 1];
                        rgb[i * 3 + 2] = overlayColor[i * 4 + 2];
                        alpha[i] = overlayColor[i * 4 + 3];
                    }
                    overlayRef.current.instanceColor.needsUpdate = true;
                    opacityAttrRef.current.needsUpdate = true;
                }
            } else {
                overlayRef.current.count = 0;
            }

            // 3. MCTS
            if ((viewMode === 'mcts' || viewMode === 'combined') && wireframeRef.current) {
                if (mctsMatrix && mctsColor) {
                    const count = mctsMatrix.length / 16;
                    wireframeRef.current.count = count;
                    if (count > 0) {
                        if (!wireframeRef.current.instanceColor || wireframeRef.current.instanceColor.count !== 10000) {
                            wireframeRef.current.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(10000 * 3), 3);
                        }
                        wireframeRef.current.instanceMatrix.array.set(mctsMatrix);
                        wireframeRef.current.instanceMatrix.needsUpdate = true;
                        wireframeRef.current.instanceColor.array.set(mctsColor);
                        wireframeRef.current.instanceColor.needsUpdate = true;
                    }
                } else {
                    wireframeRef.current.count = 0;
                }
            } else if (wireframeRef.current) {
                wireframeRef.current.count = 0;
            }
        }
    }, [step, nextStep, viewMode]);

    return (
        <group>
            {/* Opaque Structure */}
            <instancedMesh ref={opaqueRef} args={[undefined, undefined, 20000]}>
                <boxGeometry args={[0.9, 0.9, 0.9]} />
                <meshStandardMaterial roughness={0.5} metalness={0.5} emissive="#222222" />
            </instancedMesh>

            {/* Policy Overlay */}
            <instancedMesh ref={overlayRef} args={[undefined, undefined, 20000]}>
                <boxGeometry args={[0.92, 0.92, 0.92]} />
                <meshStandardMaterial
                    roughness={0.3}
                    metalness={0.2}
                    transparent={true}
                    depthWrite={false}
                />
            </instancedMesh>

            {/* MCTS Wireframes */}
            <instancedMesh ref={wireframeRef} args={[undefined, undefined, 10000]}>
                <boxGeometry args={[0.95, 0.95, 0.95]} />
                <meshBasicMaterial wireframe color="white" transparent opacity={0.5} />
            </instancedMesh>
        </group>
    );
};

export const LoadVector = ({ step }: { step: any }) => {
    if (!step) return null;

    const [C, D, H, W] = step.tensor.shape;
    if (C < 7) return null; // Need 7 channels

    const loadPoints: { pos: THREE.Vector3; dir: THREE.Vector3; magnitude: number }[] = [];
    const spatialSize = D * H * W;
    const tensorData = step.tensor.data;

    for (let d = 0; d < D; d++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const flatIdx = d * (H * W) + h * W + w;

                const fx = tensorData[CHANNEL.FORCE_X * spatialSize + flatIdx];
                const fy = tensorData[CHANNEL.FORCE_Y * spatialSize + flatIdx];
                const fz = tensorData[CHANNEL.FORCE_Z * spatialSize + flatIdx];

                const magnitude = Math.sqrt(fx * fx + fy * fy + fz * fz);

                if (magnitude > 0.01) {
                    const pos = new THREE.Vector3(d - D / 2, h + 0.5, w - W / 2);
                    const dir = new THREE.Vector3(fx, fy, fz).normalize();
                    loadPoints.push({ pos, dir, magnitude });
                }
            }
        }
    }

    if (loadPoints.length === 0) return null;
    const arrowLength = 5;
    return (
        <group>
            {loadPoints.map((lp, i) => {
                const origin = lp.pos.clone().sub(lp.dir.clone().multiplyScalar(arrowLength));
                const mid = origin.clone().add(lp.dir.clone().multiplyScalar(arrowLength / 2));
                const tip = origin.clone().add(lp.dir.clone().multiplyScalar(arrowLength));
                const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), lp.dir);

                return (
                    <group key={i}>
                        {/* Shaft */}
                        <mesh position={mid} quaternion={quat}>
                            <cylinderGeometry args={[0.05, 0.05, arrowLength, 8]} />
                            <meshStandardMaterial color="#ff6b00" />
                        </mesh>
                        {/* Tip */}
                        <mesh position={tip} quaternion={quat} translateY={-0.25}>
                            <coneGeometry args={[0.15, 0.5, 8]} />
                            <meshStandardMaterial color="#ff6b00" />
                        </mesh>
                        {/* Base Voxel Highlight */}
                        <mesh position={lp.pos}>
                            <boxGeometry args={[0.95, 0.95, 0.95]} />
                            <meshStandardMaterial color="#ff6b00" transparent opacity={0.6} />
                        </mesh>
                    </group>
                );
            })}
        </group>
    );
};

export const SupportVoxels = ({ step }: { step: any }) => {
    const supportRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);

    useEffect(() => {
        if (!supportRef.current || !step) return;

        const [C, D, H, W] = step.tensor.shape;
        if (C < 7) return;

        const spatialSize = D * H * W;
        const tensorData = step.tensor.data;
        let count = 0;

        for (let d = 0; d < D; d++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const flatIdx = d * (H * W) + h * W + w;

                    const maskX = tensorData[CHANNEL.MASK_X * spatialSize + flatIdx];
                    const maskY = tensorData[CHANNEL.MASK_Y * spatialSize + flatIdx];
                    const maskZ = tensorData[CHANNEL.MASK_Z * spatialSize + flatIdx];

                    const color = getSupportColor(maskX, maskY, maskZ);
                    if (!color) continue;

                    const pos = new THREE.Vector3(d - D / 2, h + 0.5, w - W / 2);
                    dummy.position.copy(pos);
                    dummy.updateMatrix();
                    supportRef.current.setMatrixAt(count, dummy.matrix);
                    supportRef.current.setColorAt(count, new THREE.Color(color));
                    count++;
                }
            }
        }

        supportRef.current.count = count;
        supportRef.current.instanceMatrix.needsUpdate = true;
        if (supportRef.current.instanceColor) supportRef.current.instanceColor.needsUpdate = true;
    }, [step]);

    return (
        <instancedMesh ref={supportRef} args={[undefined, undefined, 1000]}>
            <boxGeometry args={[0.95, 0.95, 0.95]} />
            <meshStandardMaterial transparent opacity={0.6} roughness={0.4} metalness={0.3} />
        </instancedMesh>
    );
};
