import { useState, useCallback } from 'react';
import { UploadCloud, Loader2, Sparkles, Activity, User, RefreshCcw, ChevronRight, Smile } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API = 'http://localhost:8001/api';

export default function AnalyzeView() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [drag, setDrag] = useState(false);
    const [storeInDb, setStoreInDb] = useState(true);

    const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null); };

    const handleFile = (f) => {
        if (!f || !f.type.startsWith('image/')) return;
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setResult(null);
        setError(null);
    };

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDrag(false);
        handleFile(e.dataTransfer.files[0]);
    }, []);

    const analyze = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        const fd = new FormData();
        fd.append('file', file);
        fd.append('generate_heatmap', 'true');
        fd.append('top_k', '5');
        fd.append('store_in_db', storeInDb ? 'true' : 'false');
        try {
            const { data } = await axios.post(`${API}/analyze`, fd);
            setResult(data);
        } catch (e) {
            setError(e.response?.data?.detail || e.message || 'Analysis failed. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    /* ── UPLOAD STATE ──────────────────────────────────────── */
    if (!result && !loading) return (
        <div>
            {/* Hero Section */}
            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <span className="format-tag" style={{ marginBottom: '16px', display: 'inline-block' }}>Beta v1.0</span>
                    <h1 style={{ fontSize: '40px', lineHeight: 1.1, marginBottom: '12px' }}>
                        AI Face <span className="text-gradient">Analytics</span>
                    </h1>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '16px', maxWidth: '480px', margin: '0 auto' }}>
                        Upload a face photo to detect age, gender, emotion, and facial attributes using <strong>Faster R-CNN</strong> and deep learning.
                    </p>
                </motion.div>
            </div>

            {/* Drop Zone */}
            <motion.div
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                className={`hero-upload ${drag ? 'drag-over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
            >
                <input id="file-input" type="file" accept="image/*" style={{ display: 'none' }}
                    onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />

                {preview ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
                        <img src={preview} alt="Preview" style={{ maxHeight: '280px', maxWidth: '100%', borderRadius: '16px', boxShadow: 'var(--shadow-md)' }} />
                        <p style={{ color: 'var(--text-tertiary)', fontSize: '14px' }}>Click to change image</p>
                    </div>
                ) : (
                    <>
                        <div className="upload-icon-ring">
                            <UploadCloud size={28} />
                        </div>
                        <h2>Drop your image here</h2>
                        <p>or click to browse from your device</p>
                        <div className="format-tags">
                            {['JPEG', 'PNG', 'WEBP', 'BMP'].map(f => <span key={f} className="format-tag">{f}</span>)}
                        </div>
                    </>
                )}
            </motion.div>

            {preview && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>

                    {/* Store in DB Toggle */}
                    <div className="card" onClick={() => setStoreInDb(!storeInDb)} style={{ cursor: 'pointer', padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '12px', border: '1px solid var(--border-color)', borderRadius: '12px', background: storeInDb ? 'rgba(99, 102, 241, 0.05)' : 'transparent', transition: 'all 0.2s' }}>
                        <div style={{ width: '40px', height: '20px', background: storeInDb ? 'var(--accent-light)' : '#444', borderRadius: '10px', position: 'relative', transition: 'background 0.3s' }}>
                            <motion.div
                                animate={{ x: storeInDb ? 22 : 2 }}
                                style={{ width: '16px', height: '16px', borderRadius: '50%', background: 'white', position: 'absolute', top: 2 }}
                            />
                        </div>
                        <span style={{ fontSize: '14px', fontWeight: 500, color: storeInDb ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                            Save results to searchable database
                        </span>
                    </div>

                    <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                        <button className="secondary-button" onClick={reset}>Clear</button>
                        <button className="primary-button" onClick={analyze} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Sparkles size={16} />
                            Analyze Face
                            <ChevronRight size={16} />
                        </button>
                    </div>
                </motion.div>
            )}

            {error && (
                <div className="error-box" style={{ marginTop: '32px' }}>
                    <p style={{ color: '#ef4444', fontWeight: 600, marginBottom: '8px' }}>Analysis Failed</p>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>{error}</p>
                    <button className="secondary-button" onClick={reset} style={{ marginTop: '16px' }}>Try Again</button>
                </div>
            )}

            {/* Feature Pills */}
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', marginTop: '48px', flexWrap: 'wrap' }}>
                {[
                    { icon: User, label: 'Age & Gender' },
                    { icon: Smile, label: 'Emotion Detection' },
                    { icon: Activity, label: 'Grad-CAM Heatmap' },
                    { icon: Sparkles, label: '40 Facial Attributes' },
                ].map(({ icon: Icon, label }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-surface)', border: '1px solid var(--border-color)', padding: '8px 16px', borderRadius: 'var(--radius-full)', fontSize: '13px', color: 'var(--text-secondary)' }}>
                        <Icon size={14} style={{ color: 'var(--accent-light)' }} />
                        {label}
                    </div>
                ))}
            </div>
        </div>
    );

    /* ── LOADING STATE ─────────────────────────────────────── */
    if (loading) return (
        <div className="center-state">
            <div style={{ position: 'relative', width: 80, height: 80 }}>
                <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: 'var(--gradient-primary)', opacity: 0.15, animation: 'pulse-glow 2s infinite' }} />
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Loader2 size={36} className="spin" style={{ color: 'var(--accent-light)' }} />
                </div>
            </div>
            <h3 style={{ fontSize: '20px' }}>Analyzing Face<span className="text-gradient">…</span></h3>
            <p style={{ color: 'var(--text-tertiary)', fontSize: '14px' }}>Running Faster R-CNN detection + deep feature extraction</p>
        </div>
    );

    /* ── RESULTS STATE ─────────────────────────────────────── */
    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="results-header">
                <div>
                    <h2 style={{ fontSize: '24px' }}>Analysis <span className="text-gradient">Results</span></h2>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '4px' }}>
                        <p style={{ color: 'var(--text-tertiary)', fontSize: '13px' }}>
                            {result.faces?.length ?? 0} face{result.faces?.length !== 1 ? 's' : ''} detected
                        </p>
                        <span className="format-tag" style={{ fontSize: '10px', padding: '2px 8px' }}>
                            Powered by {result.detector_used || 'Faster R-CNN'}
                        </span>
                    </div>
                </div>
                <button className="secondary-button" onClick={reset} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <RefreshCcw size={14} /> New Analysis
                </button>
            </div>

            {(result.faces || []).map((face, i) => (
                <div key={i} style={{ marginBottom: '32px' }}>
                    <div className="result-grid">

                        {/* LEFT — Image Column */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                            <div className="card" style={{ padding: '12px' }}>
                                <img src={`data:image/jpeg;base64,${result.detected_image}`} alt="Detected"
                                    style={{ width: '100%', borderRadius: '12px', display: 'block' }} />
                            </div>
                            {face.heatmap_emotion && (
                                <div className="card">
                                    <div className="card-label"><Activity size={14} /> Emotion Activation Map</div>
                                    <img src={`data:image/jpeg;base64,${face.heatmap_emotion}`}
                                        style={{ width: '100%', borderRadius: '8px' }} alt="Heatmap" />
                                    <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '8px' }}>
                                        Grad-CAM highlights regions driving emotion prediction
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* RIGHT — Info Column */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

                            {/* Demographics */}
                            <div className="card">
                                <div className="card-label"><User size={14} /> Demographics</div>
                                <div className="stat-grid">
                                    <div className="stat-box">
                                        <div className="stat-label">Age</div>
                                        <div className="stat-value">{face.age}</div>
                                        <div className="stat-sub">years old</div>
                                    </div>
                                    <div className="stat-box">
                                        <div className="stat-label">Gender</div>
                                        <div className="stat-value" style={{ fontSize: '22px', textTransform: 'capitalize' }}>{face.gender}</div>
                                        <div className="stat-sub">{(face.gender_confidence * 100).toFixed(0)}% confidence</div>
                                    </div>
                                </div>
                            </div>

                            {/* Emotion */}
                            <div className="card">
                                <div className="card-label"><Smile size={14} /> Emotion</div>
                                <div className="emotion-primary">{face.emotion}</div>
                                {face.emotion_all_scores && Object.entries(face.emotion_all_scores)
                                    .sort(([, a], [, b]) => b - a).slice(0, 5)
                                    .map(([emo, score]) => (
                                        <div key={emo} className="emo-bar">
                                            <div className="emo-row">
                                                <span>{emo}</span>
                                                <span>{(score * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="emo-track">
                                                <motion.div className="emo-fill"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${score * 100}%` }}
                                                    transition={{ duration: 0.7, ease: 'easeOut' }}
                                                />
                                            </div>
                                        </div>
                                    ))}
                            </div>

                            {/* Attributes */}
                            {face.present_attributes?.length > 0 && (
                                <div className="card">
                                    <div className="card-label"><Sparkles size={14} /> Facial Attributes</div>
                                    <div className="attr-grid">
                                        {face.present_attributes.map(attr => (
                                            <span key={attr} className="attr-chip">{attr.replace(/_/g, ' ')}</span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Matching Identities */}
                            {face.similar_images?.length > 0 && (
                                <div className="card">
                                    <div className="card-label"><User size={14} /> Matching Identities</div>
                                    <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '8px' }}>
                                        {face.similar_images.map((sim, idx) => (
                                            <div key={idx} style={{ flex: '0 0 80px', textAlign: 'center' }}>
                                                <div style={{
                                                    width: '80px', height: '80px', borderRadius: '12px',
                                                    backgroundColor: 'var(--bg-card)', overflow: 'hidden',
                                                    border: '1px solid var(--border-color)', marginBottom: '4px'
                                                }}>
                                                    {sim.storage_url ? (
                                                        <img src={sim.storage_url} alt="Similar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                                    ) : (
                                                        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                            <User size={20} style={{ color: 'var(--text-tertiary)' }} />
                                                        </div>
                                                    )}
                                                </div>
                                                <div style={{ fontSize: '10px', color: 'var(--accent-light)', fontWeight: 600 }}>
                                                    {(sim.similarity * 100).toFixed(0)}%
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                        </div>
                    </div>
                </div>
            ))}
        </motion.div>
    );
}
