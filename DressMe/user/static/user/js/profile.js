const steps = document.querySelectorAll('.step');
const contents = document.querySelectorAll('.step-content');
const nextBtns = document.querySelectorAll('.next-step');
const prevBtns = document.querySelectorAll('.prev-step');

let currentStep = 0;

function updateStep(index) {
    contents.forEach((c, i) => {
        c.classList.toggle('active', i === index);
        steps[i].classList.toggle('active', i === index);
    });
}

nextBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        if (currentStep < contents.length - 1) {
            currentStep++;
            updateStep(currentStep);
        }
    });
});

prevBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            updateStep(currentStep);
        }
    });
});