/**
 * GWLab Concept Demonstrator — Shared UI and Theme Script
 */

(function () {
  // 1. Theme Initialization
  function initTheme() {
    const urlParams = new URLSearchParams(window.location.search);
    const themeParam = urlParams.get('theme');
    const savedTheme = localStorage.getItem('gwlab_concept_theme');
    
    let activeTheme = 'light';
    if (themeParam === 'dark' || themeParam === 'light') {
      activeTheme = themeParam;
    } else if (savedTheme === 'dark' || savedTheme === 'light') {
      activeTheme = savedTheme;
    }
    
    document.documentElement.setAttribute('data-theme', activeTheme);
    updateThemeToggleButtons(activeTheme);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('gwlab_concept_theme', next);
    updateThemeToggleButtons(next);
    // Dispatch custom event for SVG redraws if necessary
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
  }

  function updateThemeToggleButtons(theme) {
    const buttons = document.querySelectorAll('.theme-toggle-btn');
    buttons.forEach((btn) => {
      btn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
      btn.setAttribute('title', `Active: ${theme} mode (Click to toggle)`);
      const icon = btn.querySelector('.theme-icon');
      if (icon) {
        icon.textContent = theme === 'dark' ? '☀️' : '🌙';
      }
    });
  }

  // 2. Format numbers
  window.formatFloat = function(val, decimals = 4) {
    if (isNaN(val)) return "NaN";
    if (!isFinite(val)) return val > 0 ? "+∞" : "−∞";
    const isNeg = val < 0;
    const absVal = Math.abs(val);
    const formatted = absVal.toFixed(decimals);
    return (isNeg ? "−" : "") + formatted;
  };

  window.formatScientific = function(val, sigFigs = 3) {
    if (val === 0) return "0";
    if (isNaN(val)) return "NaN";
    if (!isFinite(val)) return val > 0 ? "+∞" : "−∞";
    const isNeg = val < 0;
    const absVal = Math.abs(val);
    if (absVal >= 1e-4 && absVal < 1e4) {
      return (isNeg ? "−" : "") + absVal.toFixed(sigFigs);
    }
    const expStr = absVal.toExponential(sigFigs);
    const [mantissa, exponent] = expStr.split("e");
    const expInt = parseInt(exponent, 10);
    const superscripts = {
      "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
      "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
    };
    const superExp = String(expInt).split("").map(c => superscripts[c] || c).join("");
    return `${isNeg ? "−" : ""}${mantissa} × 10${superExp}`;
  };

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
  } else {
    initTheme();
  }

  window.toggleTheme = toggleTheme;
})();
