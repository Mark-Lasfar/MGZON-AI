function copyCode(button) {
    const code = button.previousElementSibling.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => button.textContent = 'Copy', 2000);
    });
}

document.getElementById('chatbot-link').addEventListener('click', (e) => {
    e.preventDefault();
    window.location.href = '/gradio';
});
