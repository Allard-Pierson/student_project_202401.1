import React, { useEffect } from "react";

export function Chathistory() {
  const chatHistory: string[] = ['Gebruiker: Hallo!', 'Systeem: Welkom bij de boekenzoeker. Hoe kan ik u helpen?'];

  function displayChatHistory(): void {
    const chatHistoryDiv = document.getElementById('chatHistory');
    if (!chatHistoryDiv) return;

    chatHistoryDiv.innerHTML = chatHistory.join('<br>');
  }

  useEffect(() => {
    displayChatHistory();
  }, []);

  function adjustSearchWidth(width: string): void {
    // Add logic to adjust search width
    console.log(`Adjust search width: ${width}`);
  }

  return (
    <section id="chatHistoryContainer">
      <section id="chatHistory">
        <section id="searchWidthAdjust">
          <button className="adjustButton" onClick={() => adjustSearchWidth('wider')}>Breder zoeken</button>
          <button className="adjustButton" onClick={() => adjustSearchWidth('narrower')}>Nauwer zoeken</button>
        </section>
      </section>
    </section>
  );
}
