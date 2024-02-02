import React, { useState } from "react";
import { Searchsection } from "./SearchSection";
import { Listsection } from "./ListSection";
import { Chathistory } from "./ChatHistory";

export function MiddelSection() {
  const [searchQuery, setSearchQuery] = React.useState("");

  function handleSearchInputChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
    setSearchQuery(event.target.value);
  }

  return (
    <section className="contentContainer">
      <Searchsection onSearchInputChange={handleSearchInputChange} />
      <Listsection searchQuery={searchQuery} />
    </section>
  );
}
