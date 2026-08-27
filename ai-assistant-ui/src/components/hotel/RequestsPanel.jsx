import { useEffect, useState } from "react"

export default function RequestsPanel({client}) {

  const [requests, setRequests] = useState([])

  const [filter, setFilter] = useState("all")

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const hotelName = client || "KANHAIYA"

  /* -----------------------------------
      FETCH REQUESTS
  ----------------------------------- */

  const fetchRequests = async () => {

    try {
      setError(null)

      const res = await fetch(
        "http://localhost:8000/requests"
      )

      if (!res.ok) {
        throw new Error("Failed to fetch requests")
      }

      const data = await res.json()

      console.log(
        "REQUEST DATA:",
        data
      )

      setRequests(data)

    } catch (err) {

      setError(
        "Unable to connect to the service request system."
      )
    }finally{
      setLoading(false)
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

  const updateRequestStatus = async (requestId, status) => {

    try {

      const res = await fetch(
        "http://localhost:8000/request-status",
        {

          method: "PATCH",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({

            request_id: requestId,

            status: status

          })

        }
      )

      const data = await res.json()


      if (!res.ok || !data.success) {

        throw new Error(
          "Failed to update request"
        )

      }

      fetchRequests()

    } catch (err) {

      console.error(
        "Failed to complete request:",
        err
      )

    }

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

    const getStatusLabel = (status) =>{
      const labels = {
        pending : "Pending",
        in_progress: "In Progress",
        completed: "Completed"
      }

      return labels[status] || status
    }

  // const audio =
  //   new Audio("/notification.mp3")

  // audio.play()




  /* -----------------------------------
      UI
  ----------------------------------- */

  return (

    <div className="requests-page">

      

      {/* HEADER */}
      {/* LEFT SIDE */}

      <div className="requests-header">

        <div className="requests-title">
          <p className="dashboard-label">
            OPERATIONS DASHBOARD
          </p>

          <h1>Service Requests</h1>

          <p>
            Track and manage live guest requests
          </p>
        </div>


        {/* HOTEL BRANDING */}
        {/* RIGHT SIDE */}

        <div className="header-actions">


          <div className="hotel-brand">

            <div className="hotel-brand-icon">
              {hotelName.charAt(0).toUpperCase()}
            </div>

            <div className="hotel-brand-content">

              <span className="hotel-brand-name">
                {hotelName.toUpperCase()}
              </span>

              <span className="hotel-brand-subtitle">
                Guest Service Operations
              </span>

            </div>

          </div>

          {/* LIVE STATUS */}
          <div className="live-indicator">
            <span className="live-dot"></span>
              Live Updates
          </div>
        

        </div>

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

                {req.status === "in_progress" && (
                  <span className="status-dot"></span>
                )}
                
                {getStatusLabel(req.status)}
              </span>

            </div>

            

            <p className="request-label">
              GUEST REQUEST
            </p>

            <h2 className="request-title">
              {req.display_text}
            </h2>

            <div className="request-room">
              🛏️ Room {req.room}
            </div>

            <p className="request-time">
              🕒 Received at {req.time}
            </p>

            {/* BUTTON */}

            < div className = "request-action">

              {req.status === "pending" && (

                <button
                  className="start-btn"
                  onClick={() =>
                    updateRequestStatus(
                      req.request_id,
                      "in_progress"
                    )
                  }
                > 
                  Start Service
                </button>

              )}



              {req.status === "in_progress" && (

                <button
                  className="complete-btn"
                  onClick={() =>
                    updateRequestStatus(
                      req.request_id,
                      "completed"
                    )
                  }
                >
                  Mark Complete
                </button>

              )}


              {req.status === "completed" && (
 
                <div className="completed-state">

                  <span className="completed-check">
                    ✓
                  </span>

                    Service Completed

                </div>

              )}

              </div>
            </div>

          ))}

      </div>

      {loading && (
        <div className="dashboard-state">
          Loading service requests...
        </div>
      )}

      {error && (
        <div className="dashboard-error">
        {error}
        </div>
      )}

    </div>
  )
}