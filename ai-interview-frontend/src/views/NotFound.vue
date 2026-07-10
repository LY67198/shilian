<template>
  <div class="not-found">
    <div class="error-code">404</div>
    <div class="error-msg">页面走丢了</div>
    <router-link to="/dashboard" class="back-link">返回首页</router-link>
  </div>
</template>

<script setup>
</script>

<style scoped>
.not-found {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
  gap: 16px;
}
.error-code { font-size: 96px; font-weight: 700; color: #4f46e5; letter-spacing: -4px; }
.error-msg { font-size: 18px; color: #6b7280; }
.back-link { padding: 10px 24px; border-radius: 10px; background: #4f46e5; color: white; text-decoration: none; font-weight: 600; transition: all 0.3s; }
.back-link:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(79,70,229,0.3); }
</style>