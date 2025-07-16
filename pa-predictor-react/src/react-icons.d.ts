declare module 'react-icons/*' {
  import { ComponentType } from 'react';
  
  interface IconProps {
    className?: string;
    size?: string | number;
    color?: string;
    title?: string;
  }
  
  const Icon: ComponentType<IconProps>;
  export default Icon;
} 