import { useEffect, useState } from "react"

export default function ExplorePanel({ client }) {

  const [places, setPlaces] = useState([])

  useEffect(() => {

    loadPlaces()

  }, [client])


  const loadPlaces = async () => {

    const res = await fetch(
      `http://localhost:8000/explore/${client}`
    )

    const data = await res.json()

    setPlaces(data.places || [])
  }


  return (

    <div className="explore-panel">

      <h2>Explore Nearby</h2>

      {places.map((place, i) => (

        <div
          key={i}
          className="place-card"
        >

          <div className="explore-image-wrapper">
            <img
              src={place.image}
              alt={place.name}
              className="explore-image"
            />
          </div>

          <h3>{place.name}</h3>

          <p>{place.description}</p>

          <p>📍 {place.distance}</p>

          <p>🕒 {place.timing}</p>

          <button
            onClick={() =>
              window.open(place.maps, "_blank")
            }
          >
            Open Maps
          </button>

        </div>
      ))}

    </div>
  )
}