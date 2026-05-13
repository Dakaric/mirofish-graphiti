/**
 * Temporärer Speicher für hochzuladende Dateien und Anforderungen.
 * Wird auf der Startseite nach Klick auf "Engine starten" befüllt,
 * sodass sofort weitergeleitet werden kann und der API-Call erst auf der Process-Seite stattfindet.
 */
import { reactive } from 'vue'

const state = reactive({
  files: [],
  simulationRequirement: '',
  isPending: false
})

export function setPendingUpload(files, requirement) {
  state.files = files
  state.simulationRequirement = requirement
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    simulationRequirement: state.simulationRequirement,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.simulationRequirement = ''
  state.isPending = false
}

export default state
