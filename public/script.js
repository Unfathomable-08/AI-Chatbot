function updateReadme() {
    const readmeSpan = document.querySelector("#readme-button span");
    const welcomeScreen = document.querySelector("#welcome-screen img");

    if (readmeSpan) {
        readmeSpan.textContent = "Backstory";
    } else {
        setTimeout(updateReadme, 500); // Retry after 500ms
    }

}
updateReadme();

function updateScreen() {
    const welcomeScreen = document.querySelector("#welcome-screen img");

    if (welcomeScreen) {
        welcomeScreen.src = "/public/background.png";
        welcomeScreen.classList.add("w-[400px]")
    } else {
        setTimeout(updateScreen, 500); // Retry after 500ms
    }

}
updateScreen();

function updateLogo() {
    const logo = document.querySelector(".ai-message img");

    if (logo) {
        logo.src = "/public/background.png";
        setTimeout(updateLogo, 500); // Retry after 500ms
    } else {
        setTimeout(updateLogo, 500); // Retry after 500ms
    }

}
updateLogo();