const themeToggle = document.getElementById('theme-toggle');

themeToggle.addEventListener('change', function () {
    const htmlElement = document.documentElement;

    if (themeToggle.checked) {
        // Enable dark mode
        htmlElement.classList.add('dark-mode');
        htmlElement.style.setProperty('--bg-color', '#141413');
        htmlElement.style.setProperty('--text-color', '#f0efea');
    } else {
        // Enable light mode
        htmlElement.classList.remove('dark-mode');
        htmlElement.style.setProperty('--bg-color', '#f0efea');
        htmlElement.style.setProperty('--text-color', '#141413');
    }
});