import { useState, useCallback } from 'react';
import { Search, SlidersHorizontal, UploadCloud, Loader2, User, Heart, Image } from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

const API = 'http://localhost:8000/api';

export default function SearchView() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    // Filters
    const [gender, setGender] = useState('');
    const [emotion, setEmotion] = useState('');
    const [ageMin, setAgeMin] = useState('');
    const [ageMax, setAgeMax] = useState('');

    const handleFile = (f) => { setFile(f); setPreview(URL.createObjectURL(f)); };

    const handleSearch = async () => {
        setLoading(true);
        setError(null);
        setResults(null);
        const fd = new FormData();
        if (file) fd.append('file', file);
        if (gender) fd.append('gender', gender);
        if (emotion) fd.append('emotion', emotion);
        if (ageMin) fd.append('age_min', ageMin);
        if (ageMax) fd.append('age_max', ageMax);
        fd.append('top_k', '12');
        try {
            const { data } = await axios.post(`${API}/search`, fd);
            setResults(data.results || []);
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            {/* Page Header */}
            <div style={{ marginBottom: '32px' }}>
                <h2 style={{ fontSize: '28px', marginBottom: '8px' }}>
                    Find <span className="text-gradient">Similar Faces</span>
                </h2>
                <p style={{ color: 'var(--text-tertiary)', fontSize: '14px' }}>
                    Search your indexed database using facial attributes and embedding similarity
                </p>
            </div>

            {/* Search Panel */}
            <div className="card" style={{ marginBottom: '24px' }}>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px', alignItems: 'start' }}>

                    {/* Image Upload */}
                    <div>
                        <div className="card-label"><Image size={14} /> Query Image <span style={{ fontWeight: 400, color: 'var(--text-tertiary)', textTransform: 'none', letterSpacing: 0 }}>(optional)</span></div>
                        <div
                            className="hero-upload"
                            style={{ padding: preview ? '12px' : '24px', minHeight: '140px' }}
                            onClick={() => document.getElementById('search-input').click()}
                            onDragOver={e => e.preventDefault()}
                            onDrop={e => { e.preventDefault(); e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]); }}
                        >
                            <input id="search-input" type="file" accept="image/*" style={{ display: 'none' }}
                                onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />
                            {preview ? (
                                <img src={preview} alt="Query" style={{ maxHeight: '200px', borderRadius: '8px' }} />
                            ) : (
                                <>
                                    <UploadCloud size={24} style={{ color: 'var(--accent-light)' }} />
                                    <p style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>Upload face</p>
                                </>
                            )}
                        </div>
                        {preview && (
                            <button className="secondary-button" style={{ width: '100%', marginTop: '8px', padding: '8px' }}
                                onClick={() => { setFile(null); setPreview(null); }}>
                                Clear
                            </button>
                        )}
                    </div>

                    {/* Filters */}
                    <div>
                        <div className="card-label"><SlidersHorizontal size={14} /> Attribute Filters</div>
                        <div className="filter-row">
                            <div className="filter-group">
                                <label>Gender</label>
                                <select className="filter-input" value={gender} onChange={e => setGender(e.target.value)}>
                                    <option value="">Any gender</option>
                                    <option value="male">Male</option>
                                    <option value="female">Female</option>
                                </select>
                            </div>
                            <div className="filter-group">
                                <label>Emotion</label>
                                <select className="filter-input" value={emotion} onChange={e => setEmotion(e.target.value)}>
                                    <option value="">Any emotion</option>
                                    {['happy', 'neutral', 'sad', 'angry', 'surprise', 'fear', 'disgust'].map(e => (
                                        <option key={e} value={e} style={{ textTransform: 'capitalize' }}>{e}</option>
                                    ))}
                                </select>
                            </div>
                            <div className="filter-group">
                                <label>Min Age</label>
                                <input className="filter-input" type="number" placeholder="0" min="0" max="120"
                                    value={ageMin} onChange={e => setAgeMin(e.target.value)} />
                            </div>
                            <div className="filter-group">
                                <label>Max Age</label>
                                <input className="filter-input" type="number" placeholder="100" min="0" max="120"
                                    value={ageMax} onChange={e => setAgeMax(e.target.value)} />
                            </div>
                        </div>

                        <button className="primary-button" onClick={handleSearch} disabled={loading}
                            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                            {loading ? (
                                <><Loader2 size={16} className="spin" /> Searching...</>
                            ) : (
                                <><Search size={16} /> Search Database</>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Error */}
            {error && (
                <div className="error-box">
                    <p style={{ color: '#ef4444', fontWeight: 600 }}>Search Failed</p>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '4px' }}>{error}</p>
                </div>
            )}

            {/* Empty / No Results */}
            {results !== null && results.length === 0 && (
                <div className="center-state">
                    <Search size={40} style={{ color: 'var(--text-tertiary)' }} />
                    <h3 style={{ color: 'var(--text-secondary)' }}>No matches found</h3>
                    <p style={{ color: 'var(--text-tertiary)', fontSize: '14px' }}>
                        Try adjusting your filters or{' '}
                        <a href="#" style={{ color: 'var(--accent-light)' }}>building the database</a> first.
                    </p>
                </div>
            )}

            {/* Results Grid */}
            {results !== null && results.length > 0 && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                        <h3 style={{ fontSize: '18px' }}>
                            Found <span className="text-gradient">{results.length} matches</span>
                        </h3>
                        <span style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>Sorted by similarity</span>
                    </div>

                    <div className="result-card-grid">
                        {results.map((res, i) => (
                            <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.04 }} className="result-card">
                                <div className="result-card-img">
                                    <Image size={24} />
                                </div>
                                <div className="result-card-body">
                                    {res.similarity != null && (
                                        <>
                                            <div className="score-bar">
                                                <div className="score-fill" style={{ width: `${Math.min(res.similarity * 100, 100)}%` }} />
                                            </div>
                                            <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '8px' }}>
                                                {(res.similarity * 100).toFixed(1)}% match
                                            </p>
                                        </>
                                    )}
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                        {res.gender && <span className="attr-chip" style={{ fontSize: '11px', padding: '2px 8px' }}>{res.gender}</span>}
                                        {res.age && <span className="attr-chip" style={{ fontSize: '11px', padding: '2px 8px' }}>{res.age}y</span>}
                                        {res.emotion && <span className="attr-chip" style={{ fontSize: '11px', padding: '2px 8px', textTransform: 'capitalize' }}>{res.emotion}</span>}
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            )}

            {/* Empty State — before first search */}
            {results === null && !loading && (
                <div className="center-state" style={{ paddingTop: '48px' }}>
                    <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-color)', padding: '40px', borderRadius: 'var(--radius-xl)', textAlign: 'center', maxWidth: '480px' }}>
                        <Search size={40} style={{ color: 'var(--text-tertiary)', marginBottom: '16px' }} />
                        <h3 style={{ marginBottom: '8px' }}>Hybrid Face Search</h3>
                        <p style={{ color: 'var(--text-tertiary)', fontSize: '14px', lineHeight: 1.6 }}>
                            Combine attribute filters with visual similarity — powered by FAISS vector embeddings and SQLite metadata.
                            Set your filters above and click <strong style={{ color: 'var(--text-secondary)' }}>Search Database</strong>.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
