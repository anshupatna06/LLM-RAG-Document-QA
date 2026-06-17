import { useEffect, useState } from "react"

export default function BranchesPanel({ client }) {

  const [branches, setBranches] = useState([])

  useEffect(() => {

    loadBranches()

  }, [client])


  const loadBranches = async () => {

    const res = await fetch(
      `https://llm-rag-document-qa-3.onrender.com/branches/${client}`
    )

    const data = await res.json()

    console.log("BRANCHES:", data)

    setBranches(data.branches || [])
  }


  return (

    <div className="branches-panel">

      <h2>Our Other Branches</h2>

      {branches.map((branch, i) => (

        <div
          key={i}
          className="branch-card"
        >

          <img
            src={branch.image}
            alt={branch.name}
            width="250"
          />

          <h3>{branch.name}</h3>

          <p>{branch.location}</p>

          <p>{branch.description}</p>

          <p>📞 {branch.phone}</p>

          <div>

            {/* CALL */}
            <button
              onClick={() =>
                window.open(`tel:${branch.phone}`)
              }
            >
              Call
            </button>

            {/* WHATSAPP */}
            <button
              onClick={() => {

                const clean =
                  branch.whatsapp.replace("+", "")

                window.open(
                  `https://wa.me/${clean}`,
                  "_blank"
                )
              }}
            >
              WhatsApp
            </button>

          </div>

        </div>
      ))}

    </div>
  )
}