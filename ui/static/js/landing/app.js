import { initAll } from './diffusion-panels.js';
import { drawLogo, drawLogo2, drawLogo4, drawLogoOrange } from './logos.js';
import { SDF_TRAJECTORY, loadMolecularData } from './molecular-data.js';
import { buildHeroScene, loadHeroTrajectorySources } from './hero-scene.js';
import { buildDemoScene } from './demo-scene.js';

// Run after DOM ready & layout settled
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { setTimeout(initAll, 150); setTimeout(drawLogo, 50); setTimeout(drawLogo2, 50); setTimeout(drawLogoOrange, 50); setTimeout(drawLogo4, 50); });
} else {
  setTimeout(initAll, 150); setTimeout(drawLogo, 50); setTimeout(drawLogo2, 50); setTimeout(drawLogoOrange, 50); setTimeout(drawLogo4, 50);
}

// ======= INITIALIZATION =======
let heroScene = null;
let demoScene = null;

async function initApp() {
  log('Starting app initialization...');

  try {
    await loadMolecularData();
  } catch (error) {
    log('Showcase data unavailable: ' + error.message);
    console.error(error);
    return;
  }

  // 1. Build hero scene
  try {
    let heroTrajectorySources;
    try {
      heroTrajectorySources = await loadHeroTrajectorySources();
    } catch (trajectoryError) {
      log('Additional hero trajectories unavailable: ' + trajectoryError.message);
      console.error(trajectoryError);
      heroTrajectorySources = [{ id: 'default', sdf: SDF_TRAJECTORY }];
    }
    heroScene = buildHeroScene('heroCanvas', heroTrajectorySources);
    log('Hero scene initialized');
  } catch(e) {
    log('Hero scene error: ' + e.message);
    console.error(e);
  }

  // 2. Build demo scene
  try {
    demoScene = buildDemoScene('demoCanvas');

    // Wire up controls
    const slider = document.getElementById('timelineSlider');
    const label = document.getElementById('timelineLabel');
    const playBtn = document.getElementById('playBtn');

    if (slider && label && demoScene) {
      // Calculate t value from progress (0->999, 1->0)
      const updateFromProgress = (p) => {
        const tVal = Math.round(999 - p * 999);
        slider.value = p * 1000;
        label.textContent = 't = ' + tVal;
      };

      demoScene.setProgressCallback(updateFromProgress);

      slider.addEventListener('input', (e) => {
        const p = e.target.value / 1000;
        demoScene.setProgress(p);
        demoScene.pause();
        playBtn.textContent = '▶';
        updateFromProgress(p);
      });

      playBtn.addEventListener('click', () => {
        if (demoScene.getProgress() >= 1) {
          demoScene.reset();
        }
        const playing = demoScene.toggle();
        playBtn.textContent = playing ? '⏸' : '▶';
      });

      // Auto-play after a delay
      setTimeout(() => {
        demoScene.play();
        playBtn.textContent = '⏸';
      }, 2000);
    }

    log('Demo scene initialized');
  } catch(e) {
    log('Demo scene error: ' + e.message);
    console.error(e);
  }

  // 3. Original four panels (after short delay)
  try {
    setTimeout(() => {
      initAll();
      // Original logos also start
      if (typeof drawLogo === 'function') setTimeout(drawLogo, 50);
      if (typeof drawLogo2 === 'function') setTimeout(drawLogo2, 50);
      if (typeof drawLogoOrange === 'function') setTimeout(drawLogoOrange, 50);
      if (typeof drawLogo4 === 'function') setTimeout(drawLogo4, 50);
    }, 500);
  } catch(e) {
    log('Panels init error: ' + e.message);
  }

  // 4. Scroll reveal animations
  setupRevealAnimations();

  log('App initialization complete');
}

function setupRevealAnimations() {
  const reveals = document.querySelectorAll('.reveal');
  if (!('IntersectionObserver' in window)) {
    reveals.forEach(el => el.classList.add('active'));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -50px 0px' });

  reveals.forEach(el => observer.observe(el));
}

// Boot
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initApp, 100);
  });
} else {
  setTimeout(initApp, 100);
}

