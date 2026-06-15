export default function VoiceButton({
  listening,
  startListening
}) {

  return (

    <button
      className={`mic-btn ${
        listening ? "listening" : ""
      }`}
      onClick={startListening}
    >
      🎤
    </button>

  )
}