// Wait for DOM to be fully loaded before running animation
document.addEventListener('DOMContentLoaded', function() {
  const canvas = document.getElementById("login-canvas");
  
  if (!canvas) {
    console.error('Canvas element not found!');
    return;
  }

  const ctx = canvas.getContext("2d");
  let width, height, particles;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    
    // Create particles
    particles = Array.from({ length: 70 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: Math.random() * 2 + 0.5,
      dx: (Math.random() - 0.5) * 0.6,
      dy: (Math.random() - 0.5) * 0.6
    }));
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);
    
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(102, 126, 234, 0.4)";
      ctx.fill();

      // Update position
      p.x += p.dx;
      p.y += p.dy;

      // Bounce off edges
      if (p.x < 0 || p.x > width) p.dx *= -1;
      if (p.y < 0 || p.y > height) p.dy *= -1;
    });

    requestAnimationFrame(animate);
  }

  // Initialize
  window.addEventListener("resize", resize);
  resize();
  animate();

  console.log('Canvas animation initialized successfully!');
});