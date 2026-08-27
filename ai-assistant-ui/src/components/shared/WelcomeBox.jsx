// export default function WelcomeBox({
//   business,
//   client,
//   sendMessage
// }) {

//   const data = {
//     hotel: {
//       title: `Welcome to ${client} Hotel Assistant`,
//       description: [
//         "How can I help you!!",
//         "Ask about rooms, dining & services."
        
//       ],
//       suggestions: [
//         "Check-in time?",
//         "Food menu?"
//       ]
//     }
//   }

//   const content =
//     data[business] || data.hotel

//   return (

//     <div className="welcome-box">

//       <h2>{content.title}</h2>

//       {content.description.map((d, i) => (
//         <p key={i}>{d}</p>
//       ))}

//       <div className="welcome-suggestions">

//         {content.suggestions.map((s, i) => (

//           <button
//             key={i}
//             onClick={() => sendMessage(s)}
//           >
//             {s}
//           </button>

//         ))}

//       </div>

//     </div>
//   )
// }

export default function WelcomeBox({
  business,
  client,
  sendMessage,
  onOpenExplore
}) {

  const data = {
    hotel: {
      eyebrow: "AI-Powered Guest Experience",
      title: `Welcome to ${client}`,
      subtitle: "Your AI Guest Service Assistant",
      description:
        "Ask about your stay, hotel services, dining, or request assistance directly.",

      suggestions: [
        {
          label: "🛎️ Request Service",
          query: "I need assistance with a hotel service"
        },
        {
          label: "🍽️ Explore Dining",
          query: "What dining options are available?"
        },
        {
          label: "🕐 Check-in / Check-out",
          query: "What are the check-in and check-out timings?"
        },
        {
          label: "📍 Explore Nearby",
          action:"open-explore-panel"
        }
      ]
    }
  };

  const content = data[business] || data.hotel;

  return (
    <div className="welcome-box">

      <div className="welcome-badge">
        ✦ AI-POWERED GUEST EXPERIENCE
      </div>

      <h1 className="welcome-title">
        Welcome to
        <span className="hotel-name">{client}</span>
      </h1>

      <p className="welcome-subtitle">
        Your AI Guest Assistant
      </p>

      <p className="welcome-description">
        Ask about your stay, hotel services, dining,
        or request assistance directly.
      </p>

      <div className="welcome-suggestions">

        {content.suggestions.map((item, i) => (
          <button
            key={i}
            onClick={() => {
              if (item.action === "open-explore-panel") {
                onOpenExplore()
                return
              }

              sendMessage(item.query)

            }}
          >
            {item.label}
          </button>
        ))}

      </div>

    </div>
  );
}