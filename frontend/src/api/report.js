import service, { requestWithRetry } from './index'

/**
 * Berichtsgenerierung starten
 * @param {Object} data - { simulation_id, force_regenerate? }
 */
export const generateReport = (data) => {
  return requestWithRetry(() => service.post('/api/report/generate', data), 3, 1000)
}

/**
 * Status der Berichtsgenerierung abrufen
 * @param {string} reportId
 */
export const getReportStatus = (reportId) => {
  return service.get(`/api/report/generate/status`, { params: { report_id: reportId } })
}

/**
 * Agent-Log abrufen (inkrementell)
 * @param {string} reportId
 * @param {number} fromLine - ab welcher Zeile gelesen werden soll
 */
export const getAgentLog = (reportId, fromLine = 0) => {
  return service.get(`/api/report/${reportId}/agent-log`, { params: { from_line: fromLine } })
}

/**
 * Konsolen-Log abrufen (inkrementell)
 * @param {string} reportId
 * @param {number} fromLine - ab welcher Zeile gelesen werden soll
 */
export const getConsoleLog = (reportId, fromLine = 0) => {
  return service.get(`/api/report/${reportId}/console-log`, { params: { from_line: fromLine } })
}

/**
 * Berichtsdetails abrufen
 * @param {string} reportId
 */
export const getReport = (reportId) => {
  return service.get(`/api/report/${reportId}`)
}

/**
 * Mit dem Report-Agent chatten
 * @param {Object} data - { simulation_id, message, chat_history? }
 */
export const chatWithReport = (data) => {
  return requestWithRetry(() => service.post('/api/report/chat', data), 3, 1000)
}

/**
 * Bericht als Markdown oder PDF herunterladen.
 * Blob laden und über ein temporäres <a download>-Element speichern.
 * Umgeht den Response-Interceptor (responseType 'blob' liefert das Axios-Response-Objekt direkt).
 * @param {string} reportId
 * @param {'md'|'pdf'} format
 */
export const downloadReport = async (reportId, format = 'md') => {
  const response = await service.get(`/api/report/${reportId}/download`, {
    params: { format },
    responseType: 'blob',
  })
  const blob = response instanceof Blob ? response : (response.data || response)
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${reportId}.${format}`
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
}
