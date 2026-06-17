import { useEffect, useState } from "react"

export default function RequestsPanel() {

  const [requests, setRequests] = useState([])

  const [filter, setFilter] = useState("all")

  /* -----------------------------------
      FETCH REQUESTS
  ----------------------------------- */

  const fetchRequests = async () => {

    try {

      const res = await fetch(
        "https://llm-rag-document-qa-3.onrender.com/requests"
      )

      const data = await res.json()

      console.log(
        "REQUEST DATA:",
        data
      )

      setRequests(data)

    } catch (err) {

      console.log(
        "Failed to fetch requests",
        err
      )
    }
  }

  /* -----------------------------------
      INITIAL LOAD + AUTO REFRESH
  ----------------------------------- */

  useEffect(() => {

    fetchRequests()

    const interval = setInterval(() => {

      fetchRequests()

    }, 5000)

    return () => clearInterval(interval)

  }, [])

  /* -----------------------------------
      COMPLETE REQUEST
  ----------------------------------- */

  const completeRequest = async (requestId) => {

    await fetch(
      "https://llm-rag-document-qa-3.onrender.com/request-status",
      {

        method: "PATCH",

        headers: {
          "Content-Type": "application/json"
        },

        body: JSON.stringify({

          request_id: requestId,

          status: "completed"
        })
      }
    )

    fetchRequests()
  }

  /* -----------------------------------
      STATS
  ----------------------------------- */

  const pendingCount =
    requests.filter(
      r => r.status === "pending"
    ).length

  const completedCount =
    requests.filter(
      r => r.status === "completed"
    ).length

  /* -----------------------------------
      FILTER LOGIC
  ----------------------------------- */

  const filteredRequests =
    requests.filter(req => {

      if (filter === "all")
        return true

      return req.status === filter
    })

  const audio =
    new Audio("/notification.mp3")

  audio.play()

  /* -----------------------------------
      UI
  ----------------------------------- */

  return (

    <div className="requests-page">

      {/* HEADER */}

      <div className="empty-state">

        <h2>No Active Requests</h2>

        <p>
          New guest requests will appear here
        </p>

      </div>

      <div className="requests-header">

        <h1>Service Requests</h1>

        <p>
          Track and manage live guest requests
        </p>

      </div>

      {/* STATS */}

      <div className="stats-grid">

        <div className="stats-card">

          <h2>{pendingCount}</h2>

          <p>Pending Requests</p>

        </div>

        <div className="stats-card">

          <h2>{completedCount}</h2>

          <p>Completed Today</p>

        </div>

        <div className="stats-card">

          <h2>{requests.length}</h2>

          <p>Total Requests</p>

        </div>

      </div>

      {/* FILTERS */}

      <div className="filter-row">

        <button
          className={
            filter === "all"
            ? "active-filter"
            : ""
          }
          onClick={() => setFilter("all")}
        >
          All
        </button>

        <button
          className={
            filter === "pending"
            ? "active-filter"
            : ""
          }
          onClick={() => setFilter("pending")}
        >
          Pending
        </button>

        <button
          className={
            filter === "completed"
            ? "active-filter"
            : ""
          }
          onClick={() => setFilter("completed")}
        >
          Completed
        </button>

      </div>

      {/* REQUESTS */}

      <div className="requests-grid">

        {filteredRequests.map((req) => (

          <div
            key={req.request_id}
            className="request-card"
          >

            {/* TOP */}

            <div className="request-top">

              <span className="request-id">
                {req.request_id}
              </span>

              <span
                className={`status ${req.status}`}
              >
                {req.status}
              </span>

            </div>

            {/* ROOM */}

            <h2>
              Room {req.room}
            </h2>

            {/* REQUEST */}

            <p
              className={`request-type ${req.request}`}
            >
              {req.request}
            </p>

            {/* TIME */}

            <p className="request-time">
              {req.time}
            </p>

            {/* BUTTON */}

            {req.status === "pending" && (

              <button
                className="complete-btn"
                onClick={() =>
                  completeRequest(
                    req.request_id
                  )
                }
              >
                Mark Complete
              </button>

            )}

          </div>

        ))}

      </div>

    </div>
  )
}