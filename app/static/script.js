document.addEventListener('DOMContentLoaded', () => {
    const detailEmpty = document.getElementById('detail-empty');
    const detailContent = document.getElementById('detail-content');
    
    // Elements for detail view
    const detailTitle = document.getElementById('detail-title');
    const detailTier = document.getElementById('detail-tier');
    const detailCriticality = document.getElementById('detail-criticality');
    const detailTrend = document.getElementById('detail-trend');
    const detailProbability = document.getElementById('detail-probability');
    const detailRisky = document.getElementById('detail-risky');
    const detailTruth = document.getElementById('detail-truth');

    // Simulation controls
    const simHourDisplay = document.getElementById('sim-hour');
    const btnPlayPause = document.getElementById('btn-play-pause');
    const btnRefresh = document.getElementById('btn-refresh');
    const speedSelector = document.getElementById('speed-selector');

    let activeBedId = null;
    
    // Simulation state
    let currentHour = 1; // Start at hour 1
    let isPlaying = true;
    let simulationSpeed = parseInt(speedSelector.value); // ms
    let simulationTimer = null;

    function fetchWardSnapshot() {
        simHourDisplay.textContent = `Hour ${currentHour}`;
        
        fetch(`/api/beds?hour=${currentHour}`)
            .then(response => response.json())
            .then(data => {
                const containers = {
                    'tp': document.getElementById('beds-tp'),
                    'fp': document.getElementById('beds-fp'),
                    'tn': document.getElementById('beds-tn'),
                    'fn': document.getElementById('beds-fn')
                };
                Object.values(containers).forEach(c => { if(c) c.innerHTML = ''; });
                
                let allDischarged = true;
                
                data.forEach(bed => {
                    if (!bed.is_discharged) allDischarged = false;
                    
                    const card = document.createElement('div');
                    card.className = `bed-card ${bed.is_discharged ? 'discharged' : ''} ${bed.id === activeBedId ? 'active' : ''}`;
                    card.id = `bed-${bed.id}`;
                    
                    if (bed.is_discharged) {
                        card.innerHTML = `
                            <div class="bed-header">
                                <span class="bed-id">${bed.id}</span>
                            </div>
                            <div class="bed-crit tier-LOW">--</div>
                            <div class="tier-label tier-LOW">DISCHARGED</div>
                        `;
                    } else {
                        card.innerHTML = `
                            <div class="bed-header">
                                <span class="bed-id">${bed.id}</span>
                                <span class="trend tier-${bed.tier}">${getTrendIcon(bed.trend)}</span>
                            </div>
                            <div class="bed-prediction ${bed.is_risky ? 'pred-sepsis' : 'pred-safe'}">
                                ${bed.is_risky ? 'PREDICTS SEPSIS' : 'NO SEPSIS'}
                            </div>
                            <div class="bed-crit tier-${bed.tier}">${bed.criticality.toFixed(1)}</div>
                            <div class="tier-label tier-${bed.tier}">Criticality Score</div>
                        `;
                        card.addEventListener('click', () => loadBedDetail(bed.id));
                    }
                    
                    const targetContainer = containers[bed.category.toLowerCase()];
                    if (targetContainer) targetContainer.appendChild(card);
                });
                
                if (allDischarged) {
                    togglePlayPause(false); // Pause simulation if everyone is discharged
                }
                
                // If an active bed is selected, refresh its details too
                if (activeBedId) {
                    const activeBed = data.find(b => b.id === activeBedId);
                    if (activeBed && activeBed.is_discharged) {
                        detailEmpty.classList.remove('hidden');
                        detailContent.classList.add('hidden');
                        activeBedId = null;
                    } else {
                        loadBedDetail(activeBedId);
                    }
                }
            })
            .catch(err => {
                console.error('Error fetching beds:', err);
            });
    }

    function loadBedDetail(patientId) {
        if (activeBedId) {
            const prev = document.getElementById(`bed-${activeBedId}`);
            if (prev) prev.classList.remove('active');
        }
        activeBedId = patientId;
        const current = document.getElementById(`bed-${patientId}`);
        if (current) current.classList.add('active');

        fetch(`/api/beds/${patientId}?hour=${currentHour}`)
            .then(response => response.json())
            .then(data => {
                if (data.is_discharged) return; // Handled in snapshot
                
                detailEmpty.classList.add('hidden');
                detailContent.classList.remove('hidden');
                
                detailTitle.textContent = data.id;
                detailTier.textContent = data.tier;
                detailTier.className = `badge tier-${data.tier}`;
                
                detailCriticality.textContent = data.criticality.toFixed(1);
                detailCriticality.className = `tier-${data.tier}`;
                
                detailTrend.textContent = getTrendIcon(data.trend);
                detailTrend.className = `trend-icon tier-${data.tier}`;
                
                detailProbability.textContent = `${data.probability.toFixed(1)}%`;
                
                detailRisky.textContent = data.is_risky ? "ACTIVE ALERT" : "MONITORING";
                detailRisky.style.color = data.is_risky ? "var(--color-critical)" : "var(--color-low)";

                if (data.true_onset_hour) {
                    detailTruth.textContent = `Sepsis at Hr ${data.true_onset_hour}`;
                    detailTruth.style.color = "var(--color-critical)";
                } else {
                    detailTruth.textContent = "Non-Septic";
                    detailTruth.style.color = "var(--color-low)";
                }

                const categories = ['vitals_labs', 'demographics', 'others'];
                const uiMap = {
                    'vitals_labs': document.getElementById('detail-vitals'),
                    'demographics': document.getElementById('detail-demographics'),
                    'others': document.getElementById('detail-others')
                };
                
                categories.forEach(cat => {
                    const ul = uiMap[cat];
                    if (!ul) return;
                    ul.innerHTML = '';
                    
                    if (data.drivers[cat] && data.drivers[cat].length > 0) {
                        data.drivers[cat].forEach(driver => {
                            const li = document.createElement('li');
                            li.textContent = driver;
                            if (driver.includes('↑')) li.style.borderLeftColor = 'var(--color-high)';
                            else if (driver.includes('↓')) li.style.borderLeftColor = 'var(--color-low)';
                            ul.appendChild(li);
                        });
                    } else {
                        const li = document.createElement('li');
                        li.textContent = 'None';
                        li.className = 'empty-driver';
                        ul.appendChild(li);
                    }
                });
            })
            .catch(err => console.error(`Error fetching detail for ${patientId}:`, err));
    }

    function getTrendIcon(trendStr) {
        if (trendStr === 'rising') return '↑';
        if (trendStr === 'falling') return '↓';
        return '→';
    }

    function togglePlayPause(forcePause = false) {
        if (forcePause || isPlaying) {
            isPlaying = false;
            btnPlayPause.textContent = "Play";
            btnPlayPause.classList.remove("primary");
            clearTimeout(simulationTimer);
        } else {
            isPlaying = true;
            btnPlayPause.textContent = "Pause";
            btnPlayPause.classList.add("primary");
            simulationLoop();
        }
    }

    function simulationLoop() {
        if (!isPlaying) return;
        
        fetchWardSnapshot();
        currentHour++;
        
        simulationTimer = setTimeout(simulationLoop, simulationSpeed);
    }

    // Event Listeners
    btnPlayPause.addEventListener('click', () => togglePlayPause());
    
    btnRefresh.addEventListener('click', () => {
        currentHour = 1;
        fetchWardSnapshot();
        if (!isPlaying) togglePlayPause(false); // Optionally unpause or just refresh frame
    });
    
    speedSelector.addEventListener('change', (e) => {
        simulationSpeed = parseInt(e.target.value);
    });

    // Start simulation
    fetchWardSnapshot();
    currentHour++;
    simulationTimer = setTimeout(simulationLoop, simulationSpeed);
});
