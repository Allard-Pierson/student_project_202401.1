import React from 'react';
import ReactDOM from 'react-dom/client';
import './css/styles.css';
import reportWebVitals from './reportWebVitals';
import { BrowserRouter, Routes, Route } from "react-router-dom";

//page routes
import Chatbot from "./pages/Chatbot";

const root = ReactDOM.createRoot(
    document.getElementById('root') as HTMLElement
);

export default function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route>
                    <Route index element={<Chatbot />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

root.render(
    <App />
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
