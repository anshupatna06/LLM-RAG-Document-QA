import ActionButtons from "./ActionButtons"

export default function MessageList({
  messages,
  handleAction,
  chatEndRef
}) {

  return (

    <div className="chat-window">

      {messages.map((m, i) => (

        

        <div
          key={i}
          className={`${m.role}
          message-list`}
        >

        

          {m.content
            .split("\n")
            .map((line, idx) => (

              <div key={idx}>
                {line}
              </div>

          ))}

          {m.actions && (

            <ActionButtons
              actions={m.actions}
              handleAction={handleAction}
            />

          )}

        </div>
      ))}

      <div ref={chatEndRef} />

    </div>
  )
}