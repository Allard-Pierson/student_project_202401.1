import React, { useState, ChangeEvent } from "react";

interface SearchsectionProps {
  onSearchInputChange: (event: ChangeEvent<HTMLTextAreaElement>) => void;
}

export function Searchsection({ onSearchInputChange }: SearchsectionProps) {
  const [searchQuery, setSearchQuery] = useState("");

  const handleInputChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setSearchQuery(event.target.value);
  };

  const handleButtonClick = () => {
    onSearchInputChange({ target: { value: searchQuery } } as ChangeEvent<HTMLTextAreaElement>);
  };

  return (
    <section className="searchContainer">
      <textarea
        id="searchInput"
        placeholder="Voer steekwoorden in"
        value={searchQuery}
        onChange={handleInputChange}
      />
      <button onClick={handleButtonClick}>Search</button>
    </section>
  );
}
