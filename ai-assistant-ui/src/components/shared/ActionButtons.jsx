export default function ActionButtons({
  actions,
  handleAction
}) {

  return (

    <div className="action-buttons">

      {actions.map((a, i) => (

        <button
          key={i}
          onClick={() => handleAction(a)}
        >
          {a.label}
        </button>

      ))}

    </div>
  )
}