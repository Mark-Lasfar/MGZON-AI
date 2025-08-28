function copyCode(button) {
    const code = button.previousElementSibling.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => button.textContent = 'Copy', 2000);
    });
}

document.getElementById('chatbot-link')?.addEventListener('click', (e) => {
    e.preventDefault();
    window.location.href = '/gradio';
});

// تأثيرات للكروت
document.querySelectorAll('.feature-card, .footer-card, .news-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'scale(1.05) rotate(1deg)';
    });
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'scale(1) rotate(0deg)';
    });
});

// إغلاق/فتح الـ sidebar على الموبايل
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = '☰';
    toggleBtn.className = 'sidebar-toggle';
    document.body.prepend(toggleBtn);
    
    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('active');
    });
});
