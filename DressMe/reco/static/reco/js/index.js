document.addEventListener('DOMContentLoaded', () => {
    const menuToggle = document.getElementById('menu-toggle');
    const overlay = document.getElementById('overlay');
    const sideMenu = document.getElementById('side-menu');
    const outputArea = document.querySelector('.item:nth-child(3)');
    const createOutfitBtn = document.getElementById('create-outfit');
    const elementList = document.getElementById('element-list');

    menuToggle.addEventListener('click', () => {
        sideMenu.classList.add('active');
        overlay.classList.add('active');
    });

    overlay.addEventListener('click', () => {
        sideMenu.classList.remove('active');
        overlay.classList.remove('active');
    });

    const navBurger = document.querySelector('.side-menu .burger');
    navBurger.addEventListener('click', () => {
        sideMenu.classList.remove('active');
        overlay.classList.remove('active');
    });

        elementList.addEventListener('click', (e) => {
        if (e.target.classList.contains('element')) {
            const clone = e.target.cloneNode(true);
            clone.classList.add('copied-element');
            outputArea.appendChild(clone);
            updateButton();
        }
    });

    outputArea.addEventListener('click', (e) => {
        if (e.target.classList.contains('copied-element')) {
            e.target.remove();
            updateButton();
        }
    });

    function updateButton() {
        const count = outputArea.querySelectorAll('.copied-element').length;

        if (count >= 2) {
            createOutfitBtn.classList.add('active');
            createOutfitBtn.disabled = false;
        } else {
            createOutfitBtn.classList.remove('active');
            createOutfitBtn.disabled = true;
        }
    }
});
