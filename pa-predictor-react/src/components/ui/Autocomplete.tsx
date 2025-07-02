import React, { useState, useEffect, useRef } from 'react';
import { debounce } from 'lodash';

interface AutocompleteOption {
  value: string;
  label: string;
}

interface AutocompleteProps {
  options: AutocompleteOption[];
  onSelect: (selected: AutocompleteOption[]) => void;
  placeholder: string;
  multiple?: boolean;
}

const Autocomplete: React.FC<AutocompleteProps> = ({
  options,
  onSelect,
  placeholder,
  multiple = false
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredOptions, setFilteredOptions] = useState(options);
  const [selectedItems, setSelectedItems] = useState<AutocompleteOption[]>([]);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const debouncedSearch = debounce((term: string) => {
    const filtered = options.filter(option =>
      option.value.toLowerCase().includes(term.toLowerCase()) ||
      option.label.toLowerCase().includes(term.toLowerCase())
    );
    setFilteredOptions(filtered.slice(0, 10));
  }, 300);

  useEffect(() => {
    debouncedSearch(searchTerm);
  }, [searchTerm, options, debouncedSearch]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setSearchTerm(newValue);
    setIsOpen(true);
  };

  const handleOptionSelect = (option: AutocompleteOption) => {
    if (multiple) {
      const isAlreadySelected = selectedItems.some(item => item.value === option.value);
      let newSelectedItems: AutocompleteOption[];
      
      if (isAlreadySelected) {
        newSelectedItems = selectedItems.filter(item => item.value !== option.value);
      } else {
        newSelectedItems = [...selectedItems, option];
      }
      
      setSelectedItems(newSelectedItems);
      onSelect(newSelectedItems);
    } else {
      onSelect([option]);
      setSearchTerm(option.label);
      setIsOpen(false);
    }
  };

  const removeSelectedItem = (itemToRemove: AutocompleteOption) => {
    const newSelectedItems = selectedItems.filter(item => item.value !== itemToRemove.value);
    setSelectedItems(newSelectedItems);
    onSelect(newSelectedItems);
  };

  const isOptionSelected = (option: AutocompleteOption) => {
    return selectedItems.some(item => item.value === option.value);
  };

  return (
    <div className="relative" ref={wrapperRef}>
      <div className="relative">
        {multiple && selectedItems.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {selectedItems.map((item) => (
              <span
                key={item.value}
                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
              >
                {item.label}
                <button
                  type="button"
                  className="ml-1 inline-flex items-center justify-center w-4 h-4 rounded-full text-blue-400 hover:bg-blue-200 hover:text-blue-500 focus:outline-none"
                  onClick={() => removeSelectedItem(item)}
                >
                  <span className="sr-only">Remove</span>
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        
        <input
          type="text"
          value={searchTerm}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          placeholder={placeholder}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      {isOpen && filteredOptions.length > 0 && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
          {filteredOptions.map((option, index) => (
            <div
              key={index}
              className={`px-3 py-2 cursor-pointer hover:bg-gray-100 ${
                isOptionSelected(option) ? 'bg-blue-50' : ''
              }`}
              onClick={() => handleOptionSelect(option)}
            >
              <div className="font-medium text-gray-900">{option.value}</div>
              <div className="text-sm text-gray-500">{option.label}</div>
              {isOptionSelected(option) && (
                <div className="text-xs text-blue-600">Selected</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Autocomplete; 