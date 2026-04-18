// import UploadDocuments from "./UploadDocuments"
// import DocumentManager from "./DocumentManager"

// export default function Sidebar(){

//   return(

//     <div className="sidebar">

//       <h2>Admin Panel</h2>

//       <UploadDocuments/>

//       <DocumentManager/>

//     </div>

//   )

// }

import UploadDocuments from "./UploadDocuments"
import DocumentManager from "./DocumentManager"

export default function Sidebar({ sidebarOpen, business, setBusiness, client, setClient }) {

  

  return (
     <div className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>

      <h2>Admin Panel</h2>

      {/* Assistant */}
      <div className="section">
        <label>Assistant</label>
        <select value={business} onChange={(e)=>setBusiness(e.target.value)}>
          <option value="hotel">Hotel</option>
          <option value="clinic">Clinic</option>
          <option value="restaurant">Restaurant</option>
        </select>
      </div>

      {/* Client */}
      <div className="section">
        <label>Client ID</label>
        <input
          value={client}
          onChange={(e)=>setClient(e.target.value)}
        />
      </div>

      <UploadDocuments business={business} client={client}/>
      <DocumentManager/>

    </div>
  )
}