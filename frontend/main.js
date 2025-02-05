document.getElementById('theme-toggle').addEventListener('click', function () {
  const htmlElement = document.documentElement;

  if (htmlElement.classList.contains('dark-mode')) {
      // Switch to light mode
      htmlElement.classList.remove('dark-mode');
      htmlElement.style.setProperty('--bg-color', '#f0efea');
      htmlElement.style.setProperty('--text-color', '#141413');
  } else {
      // Switch to dark mode
      htmlElement.classList.add('dark-mode');
      htmlElement.style.setProperty('--bg-color', '#141413');
      htmlElement.style.setProperty('--text-color', '#f0efea');
  }
});