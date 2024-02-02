import React from "react";


const header = () => {
    return (
        <header className="header-section">
            <nav>
                <ul className="topnav">
                    <li><a href="./">Home</a></li>
                    <li><a href="./docs">docs</a></li>
                    <li><a href="./contact">Contact</a></li>
                    <li><a href="./about">About</a></li>
                </ul>
            </nav>
        </header>
    )
}

export default header;