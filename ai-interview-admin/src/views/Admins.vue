<template>
  <div>
    <div class="page-header">
      <h1>🛡️ 管理员管理</h1>
      <button v-if="isSuperadmin" class="btn-primary" @click="openCreate">+ 新增管理员</button>
    </div>

    <div class="card filter-bar">
      <input v-model="filter.email" placeholder="搜索邮箱..." style="width:240px" @input="debouncedSearch" />
    </div>

    <div class="card">
      <table>
        <thead>
          <tr>
            <th style="width:60px">ID</th>
            <th>邮箱</th>
            <th style="width:120px">姓名</th>
            <th style="width:100px">角色</th>
            <th style="width:80px">状态</th>
            <th style="width:170px">注册时间</th>
            <th style="width:220px">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="a in items" :key="a.id">
            <td>{{ a.id }}</td>
            <td>{{ a.email }}</td>
            <td>{{ a.first_name }} {{ a.last_name }}</td>
            <td><span :class="['badge', a.role === 'superadmin' ? 'badge-purple' : 'badge-blue']">{{ a.role === 'superadmin' ? '超级管理员' : '管理员' }}</span></td>
            <td><span :class="['badge', a.is_active ? 'badge-green' : 'badge-red']">{{ a.is_active ? '正常' : '禁用' }}</span></td>
            <td>{{ formatDate(a.created_at) }}</td>
            <td>
              <button class="btn-sm btn-primary" @click="openEdit(a)">编辑</button>
              <button class="btn-sm btn-secondary" @click="openChangePwd(a)">改密</button>
              <button v-if="isSuperadmin" class="btn-sm btn-secondary" @click="toggleActive(a)">{{ a.is_active ? '禁用' : '启用' }}</button>
              <button v-if="isSuperadmin && a.id !== currentAdminId" class="btn-sm btn-danger" @click="del(a)">删除</button>
            </td>
          </tr>
          <tr v-if="!items.length"><td colspan="7" class="empty">暂无管理员</td></tr>
        </tbody>
      </table>

      <div class="pagination" v-if="total > perPage">
        <button class="btn-sm" :disabled="page <= 1" @click="page--; reload()">上一页</button>
        <span>{{ page }} / {{ Math.ceil(total / perPage) }} (共 {{ total }} 条)</span>
        <button class="btn-sm" :disabled="page >= Math.ceil(total / perPage)" @click="page++; reload()">下一页</button>
      </div>
    </div>

    <!-- 创建/编辑弹窗 -->
    <div v-if="showModal" class="modal-mask" @click.self="closeModal">
      <div class="modal">
        <div class="modal-header">
          <h2>{{ editingId ? '编辑管理员' : '新增管理员' }}</h2>
          <button class="close-btn" @click="closeModal">×</button>
        </div>
        <div class="modal-body">
          <div class="form-item">
            <label>邮箱 *</label>
            <input v-model="form.email" placeholder="admin@ai-interview.com" type="email" />
          </div>
          <div class="form-row">
            <div class="form-item">
              <label>姓</label>
              <input v-model="form.first_name" placeholder="张" />
            </div>
            <div class="form-item">
              <label>名</label>
              <input v-model="form.last_name" placeholder="三" />
            </div>
          </div>
          <div class="form-item" v-if="!editingId">
            <label>密码 *<span class="hint-text">（至少 8 位）</span></label>
            <input v-model="form.password" type="password" placeholder="********" />
          </div>
          <div class="form-item" v-if="editingId">
            <label>状态</label>
            <select v-model="form.is_active">
              <option :value="true">启用</option>
              <option :value="false">禁用</option>
            </select>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" @click="closeModal">取消</button>
          <button class="btn-primary" @click="save" :disabled="saving">{{ saving ? '保存中...' : '保存' }}</button>
        </div>
      </div>
    </div>

    <!-- 修改密码弹窗 -->
    <div v-if="showPwdModal" class="modal-mask" @click.self="closePwdModal">
      <div class="modal" style="width:420px">
        <div class="modal-header">
          <h2>修改密码</h2>
          <button class="close-btn" @click="closePwdModal">×</button>
        </div>
        <div class="modal-body">
          <div class="form-item">
            <label>当前密码 *</label>
            <input v-model="pwdForm.current_password" type="password" placeholder="********" />
          </div>
          <div class="form-item">
            <label>新密码 *<span class="hint-text">（至少 8 位）</span></label>
            <input v-model="pwdForm.new_password" type="password" placeholder="********" />
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" @click="closePwdModal">取消</button>
          <button class="btn-primary" @click="savePwd" :disabled="savingPwd">{{ savingPwd ? '保存中...' : '确定' }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { adminApi, authApi } from '../api'
import { useAuthStore } from '../stores/auth'

const items = ref([])
const total = ref(0)
const page = ref(1)
const perPage = 20
const filter = reactive({ email: '' })

const showModal = ref(false)
const editingId = ref(null)
const saving = ref(false)
const form = reactive({ email: '', first_name: '', last_name: '', password: '', is_active: true })

const showPwdModal = ref(false)
const savingPwd = ref(false)
const pwdTargetId = ref(null)
const pwdForm = reactive({ current_password: '', new_password: '' })

const authStore = useAuthStore()
const currentAdminId = computed(() => authStore.adminId ? Number(authStore.adminId) : null)
const isSuperadmin = computed(() => authStore.isSuperadmin)

let searchTimer = null
function debouncedSearch() {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => { page.value = 1; reload() }, 300)
}

function formatDate(str) {
  if (!str) return ''
  return new Date(str).toLocaleString('zh-CN')
}

async function reload() {
  try {
    const params = { page: page.value, per_page: perPage }
    if (filter.email) params.email = filter.email
    const data = await adminApi.list(params)
    items.value = data.items || data || []
    total.value = data.total ?? items.value.length
  } catch (e) { alert(e.message) }
}

function openCreate() {
  editingId.value = null
  form.email = ''; form.first_name = ''; form.last_name = ''
  form.password = ''; form.is_active = true
  showModal.value = true
}

function openEdit(a) {
  editingId.value = a.id
  form.email = a.email; form.first_name = a.first_name || ''
  form.last_name = a.last_name || ''
  form.password = ''
  form.is_active = a.is_active
  showModal.value = true
}

function closeModal() { showModal.value = false }

async function save() {
  if (!form.email.trim() || (!editingId.value && !form.password.trim())) {
    alert('邮箱和密码必填'); return
  }
  if (!editingId.value && form.password.length < 8) {
    alert('密码至少 8 位'); return
  }
  saving.value = true
  try {
    const payload = {
      email: form.email.trim(),
      first_name: form.first_name.trim() || null,
      last_name: form.last_name.trim() || null
    }
    if (editingId.value) {
      payload.is_active = form.is_active
      if (form.password) payload.password = form.password
      await adminApi.update(editingId.value, payload)
    } else {
      payload.password = form.password
      await adminApi.create(payload)
    }
    closeModal()
    await reload()
  } catch (e) { alert('保存失败: ' + e.message) }
  finally { saving.value = false }
}

function openChangePwd(a) {
  pwdTargetId.value = a.id
  pwdForm.current_password = ''
  pwdForm.new_password = ''
  showPwdModal.value = true
}

function closePwdModal() { showPwdModal.value = false }

async function savePwd() {
  if (!pwdForm.current_password || !pwdForm.new_password) { alert('请填写完整'); return }
  if (pwdForm.new_password.length < 8) { alert('新密码至少 8 位'); return }
  savingPwd.value = true
  try {
    await adminApi.changePassword(pwdTargetId.value, {
      current_password: pwdForm.current_password,
      new_password: pwdForm.new_password
    })
    alert('密码修改成功')
    closePwdModal()
  } catch (e) { alert('修改失败: ' + e.message) }
  finally { savingPwd.value = false }
}

async function toggleActive(a) {
  try {
    await adminApi.update(a.id, { is_active: !a.is_active })
    a.is_active = !a.is_active
  } catch (e) { alert(e.message) }
}

async function del(a) {
  if (!confirm(`确认删除管理员「${a.email}」？此操作不可恢复。`)) return
  try {
    await adminApi.delete(a.id)
    await reload()
  } catch (e) { alert(e.message) }
}

onMounted(reload)
</script>

<style scoped>
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.page-header h1 { font-size: 20px; }

.filter-bar { display: flex; gap: 12px; margin-bottom: 16px; padding: 14px 16px; }

.empty { text-align: center; color: #9ca3af; padding: 30px 0; }
.pagination { display: flex; align-items: center; justify-content: center; gap: 16px; padding: 16px 0 0; font-size: 13px; color: #6b7280; }

td button + button { margin-left: 4px; }

.modal-mask { position: fixed; inset: 0; background: rgba(0,0,0,.45); display: flex; align-items: center; justify-content: center; z-index: 1000; }
.modal { background: white; border-radius: 10px; width: 520px; max-width: 92vw; max-height: 90vh; display: flex; flex-direction: column; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid #f3f4f6; }
.modal-header h2 { font-size: 16px; font-weight: 600; }
.close-btn { background: transparent; font-size: 22px; padding: 0 8px; color: #9ca3af; }
.modal-body { padding: 20px; overflow-y: auto; }
.modal-footer { padding: 14px 20px; border-top: 1px solid #f3f4f6; display: flex; justify-content: flex-end; gap: 8px; }

.form-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 14px; }
.form-item { margin-bottom: 14px; }
.form-item label { display: block; font-size: 13px; color: #374151; margin-bottom: 6px; font-weight: 500; }
.hint-text { font-size: 11px; color: #9ca3af; margin-left: 6px; font-weight: 400; }
.form-item input, .form-item select, .form-item textarea {
  width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px; outline: none; resize: vertical;
}
.form-item input:focus, .form-item textarea:focus, .form-item select:focus { border-color: #4f46e5; }

.badge-purple { background: #f3e8ff; color: #6b21a8; }
</style>