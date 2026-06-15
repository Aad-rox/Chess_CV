
function Chessboard(){

    const squares = Array.from({ length: 64 })   // an array with 64 slots


    return (
    <div className="board">
      {squares.map((square, i) => (
        <div key={i} className="square">{i}</div>
      ))}
    </div>
  )
}
export default Chessboard