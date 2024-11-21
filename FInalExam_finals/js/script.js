function navigateTo(sectionId) {
    // Hide all sections
    document.querySelectorAll("main > section").forEach(section => {
      section.style.display = "none";
    });
  
    // Show the selected section
    const selectedSection = document.getElementById(sectionId);
    if (selectedSection) {
      selectedSection.style.display = "block";
    }
  }
  
  // Initialize the page by showing the "about" section
  window.onload = function() {
    navigateTo("about");
  };
  