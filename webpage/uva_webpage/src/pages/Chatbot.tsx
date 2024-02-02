import React from "react";
import "../css/styles.css";

import Header from "../Elements/Header";
import {Footer} from "../Elements/Footer";
import { MiddelSection } from "../Elements/MiddelSection";



export default function Chatbot() {
    return (
        <>
            <Header />
                <main className="index">
                    <MiddelSection />
                </main>
            <Footer />
        </>
    )
}