import api from './request'

export function startInterview(data) {
  return api.post('/interviews/start', data)
}

export function submitAnswer(interviewId, answer) {
  return api.post(`/interviews/${interviewId}/answer`, { answer })
}

export async function submitAnswerStream(interviewId, answer, onChunk, onDone, signal) {
  const authStore = (await import('../stores/auth')).useAuthStore()
  const response = await fetch(`/api/v1/interviews/${interviewId}/answer?stream=true`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authStore.token}`
    },
    body: JSON.stringify({ answer }),
    signal
  })

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  let currentEvent = ''
  let scoreData = null
  let nextQuestionData = null

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        // Blank line = SSE dispatch boundary
        if (line === '') {
          currentEvent = ''
          continue
        }
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            switch (currentEvent) {
              case 'chunk':
                onChunk(data.content)
                break
              case 'score':
                scoreData = data
                break
              case 'next_question':
                nextQuestionData = data
                break
              case 'error':
                throw new Error(data.message || data.error || 'Unknown error')
              case 'done':
                onDone({
                  score: scoreData?.score,
                  feedback: scoreData?.feedback,
                  follow_up: scoreData?.follow_up,
                  is_finished: !nextQuestionData,
                  next_question: nextQuestionData?.question,
                  question_index: nextQuestionData?.index,
                })
                break
            }
          } catch (e) {
            if (e.message !== 'Unexpected end of JSON input') throw e
          }
        }
      }
    }
  } finally {
    try { reader.releaseLock() } catch (_) {}
  }
}

export function getReport(interviewId) {
  return api.get(`/interviews/${interviewId}/report`)
}

export function getMessages(interviewId) {
  return api.get(`/interviews/${interviewId}/messages`)
}

export function getInterviews() {
  return api.get('/interviews')
}

export function deleteInterview(interviewId) {
  return api.delete(`/interviews/${interviewId}`)
}
