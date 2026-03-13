import { useState, useEffect } from 'react';
import { Database, FolderTree, RefreshCw, HardDrive, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';
import axios from 'axios';

export default function DatabaseView() {
    const [stats, setStats] = useState(null);
    const [loadingStats, setLoadingStats] = useState(true);

    const [buildDir, setBuildDir] = useState('');
    const [building, setBuilding] = useState(false);
    const [buildResult, setBuildResult] = useState(null);
    const [error, setError] = useState(null);

    const fetchStats = async () => {
        setLoadingStats(true);
        try {
            const resp = await axios.get('http://localhost:8000/api/database/stats');
            setStats(resp.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoadingStats(false);
        }
    };

    useEffect(() => {
        fetchStats();
    }, []);

    const handleBuild = async () => {
        if (!buildDir) return;
        setBuilding(true);
        setError(null);
        setBuildResult(null);

        try {
            const resp = await axios.post('http://localhost:8000/api/build-database', {
                image_dir: buildDir,
                clear_existing: false,
                max_images: null
            });
            setBuildResult(resp.data);
            fetchStats(); // Update stats after build
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
        } finally {
            setBuilding(false);
        }
    };

    return (
        <div className="database-view" style={{ padding: '0 24px 40px', maxWidth: '1000px', margin: '0 auto' }}>

            {/* Stats Cards */}
            <h3 style={{ marginBottom: '20px' }}>System Metrics</h3>
            <div className="grid-2" style={{ marginBottom: '40px' }}>

                <div className="panel" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <div style={{ background: 'rgba(99, 102, 241, 0.1)', padding: '20px', borderRadius: '50%', color: 'var(--accent-light)' }}>
                        <Database size={32} />
                    </div>
                    <div>
                        <p className="subtitle">Indexed Faces (SQLite)</p>
                        <h2 style={{ fontSize: '36px', margin: 0 }}>
                            {loadingStats ? <RefreshCw size={24} className="upload-icon" style={{ animation: 'spin 1s linear infinite' }} /> : (stats?.total_faces || 0)}
                        </h2>
                    </div>
                </div>

                <div className="panel" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '20px', borderRadius: '50%', color: '#34d399' }}>
                        <HardDrive size={32} />
                    </div>
                    <div>
                        <p className="subtitle">FAISS Vectors</p>
                        <h2 style={{ fontSize: '36px', margin: 0 }}>
                            {loadingStats ? <RefreshCw size={24} className="upload-icon" style={{ animation: 'spin 1s linear infinite' }} /> : (stats?.faiss_vectors || 0)}
                        </h2>
                    </div>
                </div>

            </div>

            {/* Database Builder */}
            <h3 style={{ marginBottom: '20px' }}>Build Index</h3>
            <div className="panel" style={{ padding: '32px' }}>

                <div style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
                    <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', color: 'var(--text-secondary)' }}>Images Directory Path (Absolute)</label>
                        <div style={{ display: 'flex', alignItems: 'center', background: 'var(--bg-base)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px' }}>
                            <FolderTree size={18} style={{ color: 'var(--text-tertiary)', marginRight: '12px' }} />
                            <input
                                type="text"
                                placeholder="/absolute/path/to/facial/images"
                                value={buildDir}
                                onChange={e => setBuildDir(e.target.value)}
                                style={{ width: '100%', background: 'transparent', border: 'none', color: 'white', outline: 'none' }}
                            />
                        </div>
                    </div>
                </div>

                <button
                    className="primary-button"
                    onClick={handleBuild}
                    disabled={building || !buildDir}
                    style={{ width: '100%' }}
                >
                    {building ? (
                        <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                            <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} /> Indexing Faces... This may take a while.
                        </span>
                    ) : 'Start Batch Indexing'}
                </button>

                {error && <p style={{ color: '#ef4444', marginTop: '16px', textAlign: 'center' }}>{error}</p>}

                {buildResult && (
                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} style={{ marginTop: '24px', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '8px', padding: '20px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: '#10b981', marginBottom: '16px' }}>
                            <CheckCircle2 size={24} />
                            <h4 style={{ margin: 0, fontSize: '16px' }}>Indexing Complete</h4>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', textAlign: 'center' }}>
                            <div><p className="subtitle">Processed</p><b style={{ fontSize: '20px' }}>{buildResult.images_processed}</b></div>
                            <div><p className="subtitle">Faces Found</p><b style={{ fontSize: '20px', color: '#6366f1' }}>{buildResult.faces_indexed}</b></div>
                            <div><p className="subtitle">No Face</p><b style={{ fontSize: '20px', color: '#ef4444' }}>{buildResult.images_no_face}</b></div>
                            <div><p className="subtitle">New DB Total</p><b style={{ fontSize: '20px', color: '#10b981' }}>{buildResult.total_in_db}</b></div>
                        </div>
                    </motion.div>
                )}

            </div>
        </div>
    );
}
