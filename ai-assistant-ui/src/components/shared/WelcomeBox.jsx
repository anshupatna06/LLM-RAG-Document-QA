export default function WelcomeBox({
  business,
  client,
  sendMessage
}) {

  const data = {
    hotel: {
      title: `Welcome to ${client} Hotel Assistant`,
      description: [
        "Ask about rooms, dining & services."
      ],
      suggestions: [
        "Check-in time?",
        "Food menu?"
      ]
    }
  }

  const content =
    data[business] || data.hotel

  return (

    <div className="welcome-box">

      <h2>{content.title}</h2>

      {content.description.map((d, i) => (
        <p key={i}>{d}</p>
      ))}

      <div className="welcome-suggestions">

        {content.suggestions.map((s, i) => (

          <button
            key={i}
            onClick={() => sendMessage(s)}
          >
            {s}
          </button>

        ))}

      </div>

    </div>
  )
}