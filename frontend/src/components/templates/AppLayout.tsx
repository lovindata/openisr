import NavBar from "../organisms/NavBar";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function AppLayout({ children }: Props) {
  return (
    <div className="md:flex md:flex-row">
      <NavBar />
      <div className="h-full w-full">{children}</div>
    </div>
  );
}