import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { authApi } from '../api'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('admin_token') || '')
  const email = ref(localStorage.getItem('admin_email') || '')
  const role = ref(localStorage.getItem('admin_role') || '')
  const adminId = ref(localStorage.getItem('admin_id') || null)

  const isLoggedIn = computed(() => !!token.value)
  const isSuperadmin = computed(() => role.value === 'superadmin')

  function setAuth(data) {
    token.value = data.access_token
    email.value = data.email || ''
    role.value = data.role || ''
    adminId.value = data.id || null
    localStorage.setItem('admin_token', data.access_token)
    localStorage.setItem('admin_email', email.value)
    localStorage.setItem('admin_role', role.value)
    if (adminId.value) localStorage.setItem('admin_id', String(adminId.value))
  }

  async function fetchMe() {
    if (!isLoggedIn.value) return
    try {
      const info = await authApi.me()
      role.value = info.role || ''
      adminId.value = info.id || null
      email.value = info.email
      localStorage.setItem('admin_role', role.value)
      if (adminId.value) localStorage.setItem('admin_id', String(adminId.value))
      localStorage.setItem('admin_email', email.value)
    } catch (e) {
      // 401 时 axios 拦截器已自动登出，此处静默
    }
  }

  function logout() {
    token.value = ''
    email.value = ''
    role.value = ''
    adminId.value = null
    localStorage.removeItem('admin_token')
    localStorage.removeItem('admin_email')
    localStorage.removeItem('admin_role')
    localStorage.removeItem('admin_id')
  }

  return { token, email, role, adminId, isLoggedIn, isSuperadmin, setAuth, fetchMe, logout }
})
