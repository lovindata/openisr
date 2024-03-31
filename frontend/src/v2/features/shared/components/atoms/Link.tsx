import { ReactNode } from "react";

interface Props {
  href: string;
  children?: ReactNode;
}

export function Link({ href, children }: Props) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}
